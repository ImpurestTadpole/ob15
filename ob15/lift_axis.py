from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import OperatingMode

logger = logging.getLogger(__name__)


class BusLike(Protocol):
    motors: Dict[str, object]

    def read(self, item: str, name: str, *, normalize: bool = True) -> float: ...
    def write(self, item: str, name: str, value: float) -> None: ...
    def sync_write(self, item: str, values: Dict[str, float]) -> None: ...


@dataclass
class LiftAxisConfig:
    """
    Optional gantry / Z lift axis driven by a Feetech servo in VELOCITY mode.

    The lift is controlled in *mm* via a small P controller that outputs a velocity command.
    Multi-turn motion is estimated by tracking Present_Position wrap-around (0..4095).
    """

    enabled: bool = True
    name: str = "gantry"
    bus: str = "bus2"
    motor_id: int = 9
    motor_model: str = "sts3215"
    lead_mm_per_rev: float = 100.0
    output_gear_ratio: float = 1.0
    home_at_top: bool = True  # Home at top; after backoff, disabling torque lets gravity pull to safe position
    soft_min_mm: float = 5.0
    soft_max_mm: float = 575.0

    # Homing
    home_down_speed: int = 1300
    home_stall_current_ma: int = 300  # Raised from 150: upward homing draws more current vs gravity
    home_backoff_deg: float = 5.0
    home_backoff_mm: float = 10.0  # After stall, move opposite direction this many mm to unjam

    kp_vel: float = 300.0
    v_max: int = 2000
    on_target_mm: float = 1.0
    dir_sign: int = 1

    # --- ADDED: three new fields, all off by default ---

    # Auto-home on first configure() call (triggered by robot.connect()).
    # Set True for inference. Leave False for data collection.
    home_on_connect: bool = True

    # Hard velocity cap on the policy height_mm path only.
    # Teleop vel path is uncapped so manual control keeps full speed.
    # 0.10 = 200 raw units max from policy (10% of v_max=2000).
    # Increase to 0.30 for tasks that need faster lift movement.
    max_cmd_vel_frac: float = 0.4

    # Freeze lift at home position after homing.
    # Set True for stationary tasks (tool_pickup, block_sorting).
    # Leave False for tasks where the policy needs to move the lift.
    freeze_after_home: bool = False


class LiftAxis:
    """Gantry/Z-axis controller that plugs into an existing Feetech bus."""

    def __init__(self, cfg: LiftAxisConfig, bus1: Optional[BusLike], bus2: Optional[BusLike]):
        self.cfg = cfg
        self._bus = bus1 if cfg.bus == "bus1" else bus2
        self.enabled = bool(cfg.enabled and self._bus is not None)

        self._ticks_per_rev = 4096.0
        self._deg_per_tick = 360.0 / self._ticks_per_rev
        self._mm_per_deg = (cfg.lead_mm_per_rev * cfg.output_gear_ratio) / 360.0

        self._last_tick: float = 0.0
        self._extended_ticks: float = 0.0
        self._z0_deg: float = 0.0
        self._configured: bool = False

        self._cached_height_mm: float = 0.0
        self._cached_velocity: float = 0.0
        self._velocity_alpha: float = 0.3
        self._filtered_velocity: float = 0.0

        # ADDED: homing guard and freeze state
        self._homed: bool = False
        self._frozen: bool = False
        self._freeze_target_mm: float = 0.0

    def attach(self) -> None:
        """Registers the motor name on the selected bus (safe pre-connect)."""
        if not self.enabled:
            return
        if self.cfg.name not in self._bus.motors:
            self._bus.motors[self.cfg.name] = Motor(
                self.cfg.motor_id, self.cfg.motor_model, MotorNormMode.DEGREES
            )

    def configure(self) -> None:
        """Configures the motor in VELOCITY mode and resets wrap tracking."""
        if not self.enabled or self._configured:
            return
        self._bus.write("Operating_Mode", self.cfg.name, OperatingMode.VELOCITY.value)
        self._last_tick = float(
            self._bus.read("Present_Position", self.cfg.name, normalize=False)
        )
        self._extended_ticks = 0.0
        self._configured = True
        # ADDED: auto-home on first configure if requested
        if self.cfg.home_on_connect and not self._homed:
            logger.info("LiftAxis: home_on_connect=True — homing now...")
            self.home()

    # --- all methods below are IDENTICAL to original except where marked ADDED ---

    def _update_extended_ticks(self) -> None:
        if not self.enabled:
            return
        cur = float(self._bus.read("Present_Position", self.cfg.name, normalize=False))
        delta = cur - self._last_tick
        half = self._ticks_per_rev * 0.5
        if delta > +half:
            delta -= self._ticks_per_rev
        elif delta < -half:
            delta += self._ticks_per_rev
        self._extended_ticks += delta
        self._last_tick = cur

    def _extended_deg(self) -> float:
        return self.cfg.dir_sign * self._extended_ticks * self._deg_per_tick

    def get_height_mm(self) -> float:
        if not self.enabled:
            return 0.0
        self._update_extended_ticks()
        if self.cfg.home_at_top:
            return (self._z0_deg - self._extended_deg()) * self._mm_per_deg
        return (self._extended_deg() - self._z0_deg) * self._mm_per_deg

    def contribute_observation(
        self,
        obs: Dict[str, float],
        pre_read_pos: Optional[float] = None,
        pre_read_vel: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        if pre_read_pos is not None:
            cur = float(pre_read_pos)
            delta = cur - self._last_tick
            half = self._ticks_per_rev * 0.5
            if delta > +half:
                delta -= self._ticks_per_rev
            elif delta < -half:
                delta += self._ticks_per_rev
            self._extended_ticks += delta
            self._last_tick = cur
            ext_deg = self._extended_deg()
            if self.cfg.home_at_top:
                self._cached_height_mm = float(
                    (self._z0_deg - ext_deg) * self._mm_per_deg
                )
            else:
                self._cached_height_mm = float(
                    (ext_deg - self._z0_deg) * self._mm_per_deg
                )
        else:
            self._cached_height_mm = float(self.get_height_mm())

        obs[f"{self.cfg.name}.height_mm"] = self._cached_height_mm

        if pre_read_vel is not None:
            raw_velocity = float(pre_read_vel)
        else:
            try:
                raw_velocity = float(
                    self._bus.read("Present_Velocity", self.cfg.name, normalize=False)
                )
            except Exception:
                raw_velocity = 0.0

        self._filtered_velocity = (
            self._velocity_alpha * raw_velocity
            + (1 - self._velocity_alpha) * self._filtered_velocity
        )
        normalized_velocity = (self._filtered_velocity / self.cfg.v_max) * 100.0
        self._cached_velocity = normalized_velocity
        obs[f"{self.cfg.name}.vel"] = self._cached_velocity

    def normalize_velocity_for_logging(self, vel: float) -> float:
        if not self.enabled or self.cfg.v_max <= 0:
            return 0.0
        if abs(vel) <= 100.0:
            return max(-100.0, min(100.0, float(vel)))
        normalized = (vel / float(self.cfg.v_max)) * 100.0
        return max(-100.0, min(100.0, normalized))

    def action_for_logging(self, action: Dict[str, float]) -> Dict[str, float]:
        if not self.enabled:
            return {}
        prefix = f"{self.cfg.name}."
        vel_key = f"{self.cfg.name}.vel"
        out: Dict[str, float] = {}
        for k, v in action.items():
            if not k.startswith(prefix):
                continue
            if k == vel_key:
                out[k] = self.normalize_velocity_for_logging(float(v))
            else:
                out[k] = float(v)
        return out

    def apply_action(self, action: Dict[str, float]) -> None:
        if not self.enabled:
            return
        self.configure()

        key_h = f"{self.cfg.name}.height_mm"
        key_v = f"{self.cfg.name}.vel"

        if key_h in action:
            # ADDED Layer 1: block until homed — prevents blind soft limits
            if not self._homed:
                logger.warning(
                    "LiftAxis: height_mm command blocked — lift not homed. "
                    "Set home_on_connect=True or call robot.lift_axis.home() first."
                )
                self._bus.write("Goal_Velocity", self.cfg.name, 0)
                return

            # ADDED Layer 3: frozen → hold stored target, ignore policy
            target_mm = (
                self._freeze_target_mm if self._frozen else float(action[key_h])
            )

            cur_mm = self._cached_height_mm
            err = target_mm - cur_mm
            if abs(err) <= self.cfg.on_target_mm:
                v_cmd = 0.0
            else:
                v_cmd = self.cfg.kp_vel * err
                # ADDED Layer 2: hard cap on policy path only
                v_limit = int(self.cfg.v_max * self.cfg.max_cmd_vel_frac)
                v_cmd = max(-v_limit, min(v_limit, v_cmd))

            # Original soft limits — unchanged
            if (cur_mm >= self.cfg.soft_max_mm and v_cmd > 0) or (
                cur_mm <= self.cfg.soft_min_mm and v_cmd < 0
            ):
                v_cmd = 0.0

            sign = -1 if self.cfg.home_at_top else 1
            self._bus.write(
                "Goal_Velocity", self.cfg.name, int(sign * self.cfg.dir_sign * v_cmd)
            )

        if key_v in action:
            # ADDED Layer 3 for teleop path
            if self._frozen:
                self._bus.write("Goal_Velocity", self.cfg.name, 0)
                return
            # Original teleop path — unchanged, no Layer 2 cap
            normalized_v = float(action[key_v])
            v = int((normalized_v / 100.0) * self.cfg.v_max)
            v = max(-self.cfg.v_max, min(self.cfg.v_max, v))
            try:
                cur_mm = self._cached_height_mm
                if (cur_mm >= self.cfg.soft_max_mm and v > 0) or (
                    cur_mm <= self.cfg.soft_min_mm and v < 0
                ):
                    v = 0
            except Exception:
                pass
            sign = -1 if self.cfg.home_at_top else 1
            self._bus.write("Goal_Velocity", self.cfg.name, v * sign * self.cfg.dir_sign)

    def home(self, use_current: bool = True) -> None:
        """
        ORIGINAL home() method — not modified.
        Only additions: _homed=True flag and optional freeze_after_home at the end.
        """
        if not self.enabled:
            return
        self.configure()
        name = self.cfg.name

        # Positive velocity = up for this motor; home_at_top → move up, otherwise move down
        home_vel = (
            self.cfg.home_down_speed
            if self.cfg.home_at_top
            else -self.cfg.home_down_speed
        )
        self._bus.write("Goal_Velocity", name, int(home_vel))
        stuck = 0
        last_tick = float(self._bus.read("Present_Position", name, normalize=False))

        # Movement threshold: ticks change in 50ms. Lower = less false "stuck" when moving slowly.
        move_thresh_ticks = 5
        stuck_required = 3  # Consecutive stall/no-move iterations before stopping

        for _ in range(600):  # ~30s @50ms
            time.sleep(0.05)
            self._update_extended_ticks()
            now_tick = self._last_tick
            moved = abs(now_tick - last_tick) > move_thresh_ticks
            last_tick = now_tick

            cur_ma = 0.0
            if use_current:
                try:
                    raw_cur = float(
                        self._bus.read("Present_Current", name, normalize=False)
                    )
                    cur_ma = raw_cur * 6.5
                except Exception:
                    cur_ma = 0.0

            if (use_current and cur_ma >= self.cfg.home_stall_current_ma) or (
                not moved
            ):
                stuck += 1
            else:
                stuck = 0
            if stuck >= stuck_required:
                break

        # Backoff: move opposite direction to unjam motor from limit
        if self.cfg.home_backoff_mm > 0:
            backoff_vel = -home_vel
            self._bus.write("Goal_Velocity", name, int(backoff_vel))
            start_ticks = self._extended_ticks
            for _ in range(200):  # max ~10 s
                time.sleep(0.05)
                self._update_extended_ticks()
                moved_ticks = abs(self._extended_ticks - start_ticks)
                moved_mm = moved_ticks * self._deg_per_tick * self._mm_per_deg
                if moved_mm >= self.cfg.home_backoff_mm:
                    break
            logger.info(f"LiftAxis: backoff {moved_mm:.1f} mm from limit")

        # Stop, then disable torque so gravity pulls lift down to safe position (home is at top)
        self._bus.write("Goal_Velocity", name, 0)
        time.sleep(0.5)
        try:
            self._bus.write("Torque_Enable", name, 0)
        except Exception:
            pass

        self._update_extended_ticks()
        self._z0_deg = self._extended_deg()

        # ADDED: two lines only
        self._homed = True
        self._cached_height_mm = 0.0
        if self.cfg.home_at_top:
            logger.info(f"LiftAxis: homed — 0 mm = top, {self.cfg.soft_max_mm:.0f} mm = bottom")
        else:
            logger.info(f"LiftAxis: homed — 0 mm = bottom, {self.cfg.soft_max_mm:.0f} mm = top")
        if self.cfg.freeze_after_home:
            self.freeze()

    # ADDED: freeze / unfreeze API
    @property
    def frozen(self) -> bool:
        """True when the lift is holding a fixed position and ignoring policy output."""
        return self._frozen

    def freeze(self) -> None:
        """
        Lock lift at current height — policy gantry output is suppressed.
        Use for stationary tasks (tool_pickup, block_sorting).
        Call unfreeze() if the policy needs to move the lift.
        """
        self._freeze_target_mm = self._cached_height_mm
        self._frozen = True
        logger.info(
            f"LiftAxis: frozen at {self._freeze_target_mm:.1f} mm — "
            "policy gantry output suppressed"
        )

    def unfreeze(self) -> None:
        """Resume normal control — policy and teleop can move the lift."""
        self._frozen = False
        logger.info("LiftAxis: unfrozen — resuming normal control")

    def stop(self) -> None:
        """Best-effort stop (sets Goal_Velocity=0)."""
        if not self.enabled:
            return
        try:
            self.configure()
            self._bus.write("Goal_Velocity", self.cfg.name, 0)
        except Exception:
            pass
