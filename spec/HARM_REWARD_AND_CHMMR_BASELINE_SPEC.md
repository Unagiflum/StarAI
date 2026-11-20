# Harm Reward And Chmmr Baseline Specification

## Status And Authority

This document is the implementation source of truth and progress tracker for:

1. Satellite HP bars on Chmmr satellites.
2. Baseline behaviours for fighting a Chmmr that still has satellites.
3. A planet-avoidance lookahead scaled to each ship's agility.
   3A. Pre-emptible held weapons (Chenjesu, Kohr-Ah, Melnorme).
4. A discount half-life setting in seconds, replacing raw gamma.
5. A "harm" reward scheme that rewards only lasting harm to the opponent.
6. A gradual refactor of baseline strategies into shared, prioritised
   behaviours.

Phases are implemented in the order of the tracker at the end. A later phase
starts only after the earlier phase's acceptance checks pass. Phase 6 is
implemented last, one ship at a time.

Design decisions are recorded under "Decisions". The phase sections are
written to match them.

Created 2026-11-20. Decisions recorded 2026-11-20.

## Verified Game Facts

These were checked in the source on 2026-11-20 and drive the designs below.

### Chmmr satellites

- 3 satellites, 10 HP each (30 total), orbiting 256 from the Chmmr. The orbit
  phase advances one step per frame with a period of 64 frames (2.7 s), so a
  satellite passes any given bearing about every 21 frames (0.9 s).
- Each satellite fires a 1-damage laser every 3 frames (`weapon_wait` 2) at a
  target within 256.
- Target choice (`ChmmrSatellite._select_laser_target`): the opponent ship
  first, if it is in range, visible and not cloaked; otherwise the enemy object
  with the lowest HP, then the nearest.
- Satellites collide with enemy projectiles. They do not collide with ships.
- Satellites block lasers (default `blocks_lasers`).
- Satellites do not regenerate.

### Chmmr weapons

- A1 laser: range 600, 2 damage.
- A2 tractor: an impulse of 12 ÷ target mass toward the Chmmr on each use. It
  needs a visible, inertial target, so cloaked ships and non-inertial ships
  (Arilou, Slylandro) are not pulled. Its cost of 1 and battery regeneration of
  1 per 2 frames allow a use about every other frame.

### Distance rings around the Chmmr

R is the distance from the Chmmr. Ship and satellite sizes shift these numbers
by a few tens of units.

- Out of every satellite's range: R > 512.
- Also out of the Chmmr laser: R > 600.
- A weapon with reach r can hit the near satellite when R ≤ 256 + r.

### Weapons and costs for this work

| Ship | Weapon facts |
|---|---|
| Thraddash | <ul><li>A1: 2 HP shot, reach 1,620, one shot per 13 frames.</li><li>A2 flames: 1 HP, 2 damage each.</li><li>Mass 7.</li></ul> |
| Supox | <ul><li>A1: 1 HP shot, reach 1,080, one shot per 3 frames, the same rate as one satellite's zaps.</li></ul> |
| Pkunk | <ul><li>A1: 1 HP shots, reach 432, front and both side guns per press, can fire every frame.</li><li>12 battery, refilled only by A2.</li><li>Mass 1.</li></ul> |
| Arilou | <ul><li>A1: laser, reach 400. It aims itself at the opponent ship and can't be pointed elsewhere.</li><li>2 energy per shot, 20 battery.</li><li>Not inertial.</li></ul> |
| Vux | <ul><li>A1: laser, reach 644, fixed forward.</li><li>A2 limpets: 1 HP.</li><li>Mass 6.</li></ul> |

Ship costs run from 5 (Shofixti) to 30 (Chmmr, Kohr-Ah, Kzer-Za). Vmax = 30.

### Pkunk rebirth

The chance is `0.7 × 0.3^rebirths`. The expected number of deaths per Pkunk is
about 1.86, so 0.5 kill credit per death averages about 0.93 kills per Pkunk.

### Training returns

- Returns are discounted sums of reward, cut off where the discount weight falls
  below 0.01 (`discount_cutoff_frames`). There is no bootstrapping.
- The default gamma of 0.99 per frame equals a half-life of 2.9 s and a cutoff
  of 459 frames (19 s).
- The match limit is 2,400 frames (100 s). The game runs at 24 FPS.
- The `causal` reward mode already places trainee-caused positive effects on
  the frame of the action that caused them (`EVENT_REWARD_UPDATE_SPEC.md`).

## Baseline Measurement: Opponent vs Chmmr Before Changes

20 fights per ship, baseline strategy on both sides, 100 s limit, before any
change in this spec.

| Ship | W-L | Chmmr crew dealt (of 42) | Satellite HP dealt (of 30) | Avg seconds alive |
|---|---|---|---|---|
| Spathi | 0-19 | 0.6 | 0.0 | 19.8 |
| Shofixti | 0-20 | 10.9 | 23.0 | 1.6 |
| Pkunk | 0-20 | 0.1 | 0.1 | 9.8 |
| Arilou | 0-20 | 0.7 | 0.0 | 5.5 |
| ZoqFotPik | 0-20 | 3.1 | 0.1 | 3.2 |
| Supox | 0-17 | 1.1 | 0.2 | 46.1 |
| Thraddash | 0-20 | 0.6 | 0.1 | 4.2 |
| Earthling | 0-20 | 0.2 | 1.9 | 12.0 |
| Yehat | 0-20 | 1.8 | 0.7 | 16.3 |
| Mycon | 0-20 | 7.2 | 10.1 | 19.8 |
| Ilwrath | 0-20 | 0.0 | 0.0 | 11.7 |
| Androsynth | 0-20 | 4.5 | 0.2 | 14.9 |
| Druuge | 19-1 | 41.7 | 7.3 | 19.7 |
| KzerZa | 0-6 (14 draws) | 10.4 | 2.1 | 79.9 |
| Orz | 0-20 | 1.6 | 0.0 | 6.5 |
| Melnorme | 0-20 | 7.2 | 3.7 | 18.6 |
| Utwig | 0-20 | 6.3 | 1.4 | 11.0 |
| Vux | 0-20 | 19.6 | 2.1 | 5.0 |

Missing from this table: Slylandro and Syreen. Add them before Phase 2 starts.

---

## Phase 1: Satellite HP Bars

### Requirements

- Draw a horizontal bar above each live Chmmr satellite.
- Length is proportional to current HP ÷ 10. The full length is a fixed width,
  scaled with the view like other sprites.
- Colour, by HP value:

  | HP | Colour |
  |---|---|
  | 10 (full) | white `#FFFFFF` |
  | 9 | green `#00FF00` |
  | 5 | yellow `#FFFF00` |
  | 1 | red `#FF0000` |

  - Between 1 and 9, interpolate linearly: red to yellow from 1 to 5, then
    yellow to green from 5 to 9.
  - A satellite that starts below full HP uses the same table. Colour and
    length follow its HP value, not a fraction of its own starting HP.
- Rendering only. No change to gameplay, collisions, training or headless runs.
- Shown wherever satellites are drawn: normal play and the training display.

### Tests

- A pure function maps HP to (length fraction, colour). Unit-test it at 10, 9,
  7 (yellow-green midpoint), 5, 3 (red-yellow midpoint) and 1 HP.
- A satellite draw call with a dummy surface does not raise.

### Supox shot end animation

- Play the Supox A1 end animation only when the shot hits something: a ship, a
  satellite, or another object it damages.
- No animation when the shot expires or is shot down, as in the original game.
- Other projectiles are unchanged.
- Test: the end animation plays on a hit, and not on expiry or on destruction by
  a satellite laser.

---

## Phase 2: Chmmr Matchup Behaviours

All changes apply only when the opponent is a Chmmr. Other matchups are
unchanged by construction.

### Shared helpers (engine)

- `chmmr_satellites(ctx)`: the live enemy satellites from the observation, with
  position, velocity and HP. Observed objects already carry `hit_points`.
  Verify how satellites are identified in the observation (name or kind field).
- `satellite_ring(ctx, reach)`: the distance band (R_min, R_max) around the
  Chmmr.
  - R_min = 512 + margin. Use 600 + margin when staying out of the Chmmr laser
    is possible.
  - R_max = 256 + reach − margin.
  - Returns None when R_max < R_min.
- `satellite_target(ctx)`: the satellite to shoot, by default the lowest-HP live
  satellite within reach. Returns the Chmmr when no satellites are left.
- `fire_when_intercept(ctx, target)`: fire when the predicted shot meets the
  target. Lasers fire when the target lies on the laser line within reach.

### Per-ship plans

Each ship is implemented and measured on its own, in this order. Keep a change
only if it raises satellite HP or Chmmr crew dealt without shortening the
average time alive. A win isn't required.

| # | Ship | Plan | Notes and verification |
|---|---|---|---|
| 1 | Thraddash | <ul><li>Lay the A2 flame trail between itself and the Chmmr while satellites are near.</li><li>Fire A1 at the nearest satellite, or at the Chmmr once the trail is absorbing zaps.</li></ul> | <ul><li>In the original game the bait trail beats the Chmmr. Here it is 0-20, dealing 0.6 crew and dying in 4.2 s.</li><li>Investigate first by watching fights: is the Chmmr in `THRADDASH_SLOW_TURNERS`, does our baseline Chmmr follow into the trail, and does the tractor pull the Thraddash in?</li><li>Flames (1 HP) are shot before A1 shots (2 HP), so the trail works as a decoy.</li><li>A1 survives one zap.</li></ul> |
| 2 | Arilou | <ul><li>Hold R ≈ 600–656.</li><li>Fire A1 when a satellite crosses the line from the Arilou to the Chmmr.</li></ul> | <ul><li>The laser aims itself at the Chmmr, and satellites block lasers, so the crossing satellite takes the hit.</li><li>The tractor doesn't affect the Arilou.</li><li>One battery is 10 shots, one satellite.</li></ul> |
| 3 | Supox | <ul><li>Hold the satellite ring for its reach.</li><li>Stream A1 at the target satellite.</li></ul> | <ul><li>Its fire rate equals one satellite's zap rate, so success is uncertain. Its current results leave nothing to lose.</li><li>Phase 1 HP bars make hits visible.</li></ul> |
| 4 | Pkunk | <ul><li>Fly tangentially at R ≈ 600–688.</li><li>Fire continuously, side gun toward the target, as a satellite passes.</li></ul> | <ul><li>Fires up to 3× one satellite's zap rate.</li><li>The tractor pulls it hardest (mass 1).</li><li>Builds on the existing side-gun strafing.</li></ul> |
| 5 | Vux | <ul><li>No movement goal of its own against the tractor.</li><li>While within 256 of any satellite (satellites shoot the ship first), launch limpets at the Chmmr.</li><li>Laser the target satellite or the Chmmr, whichever is lined up.</li></ul> | Limpets (1 HP) are shot down only when the satellites have no ship to shoot. |
| 6 | Orz | <ul><li>Launch marines only when ≤ 1 satellite is left.</li><li>Otherwise shoot satellites with the turret.</li></ul> | Today marines launch below 2 regardless of satellites. |
| 7 | Kzer-Za | Fighters only when 0 satellites are left. | **Already implemented** (`strategies.py`, `fighter_counter`). Confirm by measurement only. |
| 8 | Androsynth | <ul><li>Blazer form at the Chmmr.</li><li>Save battery until close, then go all out.</li></ul> | Satellites don't collide with ships. Verify whether the blazer can hit satellites at all; if not, target only the Chmmr. |
| 9 | Ilwrath | <ul><li>Stay cloaked until in A1 range of the Chmmr.</li><li>Then unload A1 until one ship dies.</li></ul> | Satellites ignore cloaked ships, and the tractor needs a visible target. |
| 10 | Utwig | <ul><li>Hold the shield while in satellite range to stay charged from their lasers.</li><li>Fire A1 at the Chmmr whenever the battery allows.</li></ul> | It can't shield and fire at once. Measure crew dealt, not just wins. |
| 11 | Earthling | <ul><li>Build lateral speed so it orbits the Chmmr instead of being pulled in.</li><li>Use the A2 lasers once close.</li></ul> | <ul><li>Verify whether the A2 point-defence picks satellites. It picks enemy objects by distance.</li><li>A1 can't get through the satellites.</li></ul> |
| 12 | Umgah | Point at the Chmmr and keep the cone on. | No real hope. Verify whether the cone damages satellites. |
| 13 | Slylandro, ZoqFotPik, Spathi, Syreen, Yehat | Try the shared helpers: ring plus satellite targeting. | Keep per ship only if it helps. |

### Measurement

- Measure each ship before and after, 20–40 fights against the Chmmr. Metrics:
  - satellite HP dealt;
  - Chmmr crew dealt;
  - seconds alive;
  - W-L.
- Make the measurement script reusable: extend the baseline arena tooling to
  report satellite HP dealt. It is not part of the unit test suite.

### Tests

- Ring computation: values, and None when R_max < R_min.
- Satellite target choice: lowest HP, then nearest, then the Chmmr when none are
  left.
- Arilou line test: a satellite on the line fires; one off the line doesn't.
- Per-ship gates:
  - Orz marines only at ≤ 1 satellite.
  - Vux limpets only while a satellite is in range of the Vux.
  - Ilwrath stays cloaked until in range.
- Behaviour against non-Chmmr opponents is unchanged.

---

## Phase 3: Planet Lookahead By Agility

Replace the fixed `PLANET_AVOID_HORIZON_SECONDS = 4.0` with a per-ship horizon:

```text
horizon = turn_180_frames + accel_frames + margin_frames
```

- `turn_180_frames`: from `ShipFlight` turning rules.
- `accel_frames`: frames from 0 to `max_thrust`, from `thrust_increment` and
  `thrust_wait`. It is 0 for non-inertial ships, and for the Slylandro's
  constant speed.
- Clamp to [1 s, 4 s] until measured.

Acceptance:

- Planet deaths do not rise.
- Frames claimed by the planet rule fall for agile ships (Arilou, Pkunk, Supox).

Tests: horizon values for an agile ship, a slow ship and a non-inertial ship.

---

## Phase 3A: Pre-emptible Held Weapons

### Problem

Three ships fire weapon 1 by holding it, and can't fire again until they let
go:

| Ship | While A1 is held | On release | Today's baseline |
|---|---|---|---|
| Chenjesu | The crystal flies on. | It bursts into shards. A new crystal needs a release, then a fresh press. | <ul><li>Holds while the crystal still closes on the enemy (`_hold_while_closing_on_enemy`).</li><li>The rule 1 shoot-down includes the Chenjesu, but it presses A1. While a crystal is held, pressing only keeps holding it, so the planned new shot never fires.</li></ul> |
| Kohr-Ah | The saw flies on. | It parks where it is. A new saw needs a release, then a fresh press. | <ul><li>Holds while the saw still closes on the enemy.</li><li>Excluded from the shoot-down (`SHOOT_DOWN_EXCLUDED_SHIPS`).</li></ul> |
| Melnorme | The shot charges. | It fires at the current charge, in the nose direction. | <ul><li>Held from the start, released only when in range and pointing at the enemy.</li><li>Excluded from the shoot-down.</li></ul> |

A shot in flight or charging is kept even when something matters more:
- shooting down an incoming projectile or enemy object, such as a fighter,
  marine, mine or Dogi;
- a better shot at the enemy.

This applies especially when the held shot has little chance of hitting.

### Requirements

1. **Value of the held shot.** Estimate its expected damage to the enemy:
   - Chenjesu and Kohr-Ah: from the predicted closest approach of the shot in
     flight to the enemy, whether it is still closing, and the time to reach it.
   - Melnorme: from the current charge level and whether the enemy is in range
     along the nose.

   A shot that is receding or predicted to miss is worth about 0.
2. **Pre-emption.**
   - A competing goal may take weapon 1 when its value exceeds the held shot's
     value. Competing goals include a shoot-down from the dodge rule and a new
     shot at the enemy or another enemy object.
   - When the held shot is worth about 0, any useful goal wins.
3. **Release mechanics are modelled.**
   - A new shot needs one frame with A1 released, then a fresh press once A1 is
     ready.
   - Plans that fire a new shot (shoot-down options, the dodge rule's
     simulation) start from that release frame. They must not assume holding A1
     fires.
   - The release has its own effect, and options include it:
     - the Chenjesu crystal bursts, and its shards may themselves stop a threat;
     - the Kohr-Ah saw parks as an obstacle;
     - the Melnorme shot fires at its current charge.
4. **Melnorme release as the shot.** An option releases the charging shot
   itself at the threat or enemy once the nose points at it, without waiting
   for full charge or for the enemy to be in range.
5. **Shoot-down exclusions.**
   - Remove the Kohr-Ah and Melnorme from `SHOOT_DOWN_EXCLUDED_SHIPS` once 3 and
     4 model them correctly.
   - Keep a ship excluded if measurement shows the release costs more than it
     saves.
6. **No change for other ships.**

### Measurement

For each of the three ships, before and after:
- crew lost per minute to shots that could be shot down;
- crew dealt per minute;
- W-L.

Use opponents with projectiles or launched objects: Druuge, Kzer-Za, Orz,
Spathi, Chenjesu, Mycon and Earthling. Keep the change per ship only if crew
lost falls without crew dealt falling more.

### Tests

- Chenjesu with a crystal held and an imminent threat on a shoot-down line:
  - it releases, then presses again once ready, firing a new crystal at the
    threat;
  - with no competing goal and the crystal closing, it keeps holding.
- A receding crystal or saw is released for a new shot at the enemy.
- Kohr-Ah: the same release-then-fire sequence; the parked saw remains.
- Melnorme: it releases a partial charge at an imminent threat it points at,
  and keeps charging when nothing competes.
- The dodge rule's simulation includes the release frame for held weapons.
- Strategies for other ships are unchanged.

---

## Phase 4: Discount Half-Life Setting

### Requirements

- The Regimen setting "Gamma" becomes "Discount half-life (s)", h, with a range
  of 1–30 s and a default of 5 s.
- Derived values (FPS = 24):

  ```text
  gamma = 0.5 ** (1 / (h * FPS))
  cutoff_frames = ceil(ln(0.01) / ln(gamma)) ≈ 6.644 * h * FPS
  ```

  | Half-life | Gamma | Cutoff |
  |---|---|---|
  | 1 s | 0.97153 | 160 frames (6.6 s) |
  | 2.9 s (today's 0.99) | 0.99 | 459 frames (19 s) |
  | 5 s | 0.99424 | 798 frames (33 s) |
  | 10 s | 0.99712 | 1,595 frames (66 s) |
  | 15 s | 0.99808 | 2,392 frames (100 s) |
  | 30 s | 0.99904 | 4,784 frames (199 s) |

- At the default 100 s match limit, the cutoff exceeds the match time above
  about 15 s. Most returns then end at the match end instead of the cutoff.
  Samples also wait longer in staging before reaching replay.
- Cutoff warning:
  - Show a warning next to the setting whenever `cutoff_frames` exceeds the
    match frame length currently set on the Regimen tab.
  - Recheck it when either setting changes.
  - Example text: "Credit reaches back further than a match lasts (cutoff 33 s,
    match 30 s); returns will end at the match end."
  - The warning does not block saving or starting.
- Saved configs:
  - Store the half-life.
  - When loading an old config with only gamma, derive h = ln 0.5 ÷ (ln gamma ×
    FPS) and round it to 0.1 s for display.
- A coordinated run requires equal half-lives, as it now requires equal gamma.

### Tests

- Half-life ↔ gamma round trip.
- Cutoff frames at 1, 5 and 30 s.
- The warning appears exactly when the cutoff exceeds the match frame length,
  including after either value changes.
- Old configs load with the derived half-life.
- The coordinated mismatch check uses the half-life.

---

## Phase 5: Harm Reward Scheme

### Concept

A new reward scheme, `harm`, rewards only lasting harm to the opponent, and
optionally penalises lasting harm to the trainee. Every other component is off
in this scheme: point, range, speed, use, battery, debuff and object kills. The
baseline strategy supplies the behavioural guidance those components used to.

The existing scheme stays available as `classic`. The scheme is separate from
the existing placement mode (`legacy`, `shadow`, `causal`). Harm rewards use
causal placement as today:

- Trainee-caused harm to the opponent goes on the causing action's frame.
- Penalties stay on the frame the harm happens.
- Self-inflicted own losses, such as the Druuge crew burn, go on the trainee
  action frame that caused them.
- Opponent crew gains produce no reward or penalty.

### Settings (Rewards tab, harm scheme)

Penalty weights are set separately from reward weights, as they are today.

| Weight | Meaning | Default |
|---|---|---|
| D_enemy | Damage to the opponent | 1.28 |
| K_enemy | Kill of the opponent | 1.28 |
| D_self | Damage taken, as a penalty | 0 |
| K_self | Own death, as a penalty | 0 |

### Formulas

Symbols:

- V: the damaged ship's cost.
- s = sqrt(V / Vmax), with Vmax = 30, the highest cost in the ship catalog.
- HP: the ship's normal starting crew.
- satHP: normal total satellite HP, 30.
- has_sats: 1 for the Chmmr, otherwise 0.
- p: 0.5 for the Pkunk, otherwise 1.

The same formulas give the reward and the penalty:

- The reward uses the opponent's V, HP, satellites and p, with D_enemy and
  K_enemy.
- The penalty uses the trainee's own values, with D_self and K_self.

```text
Damage = D * s * [0.5 * p * hp_damage/HP + L + 0.5 * sat_damage/satHP] / (1 + 0.5 * has_sats)
Kill   = K * s * (ship_kills + sat_kills/3) / (1 + has_sats)
```

The reward on a frame is the change in each value since the previous frame.

Damage definitions:

- Crew means effective crew: crew on board plus launched crew that is away
  (Kzer-Za fighters, Orz marines).
  - Launching or recovering a unit doesn't change effective crew.
  - A unit lost for good is a 1-crew loss at that moment.
- Pkunk: each life is tracked separately, from that life's starting crew. A
  rebirth's crew refill is not a gain. Damage counts afresh for each life, halved
  by p.
- `hp_damage` is measured differently for the opponent and the trainee.

#### Opponent `hp_damage`: new lows caused by the trainee

The same rule applies to every opponent. For ships that can't gain crew it is
the same as plain damage.

```text
low = opponent's starting crew            (per life for the Pkunk)

on each opponent crew loss from c_before to c_after:
    new_low_amount = max(0, low - c_after)
    if the loss was caused by the trainee:
        hp_damage += new_low_amount        (placed on the causing action's frame)
    low = min(low, c_after)

on each opponent crew gain:
    nothing; low is unchanged
```

- Losses the trainee didn't cause still lower `low`, so the trainee is never
  paid for them later. Examples: planet hits, Druuge crew burn, own fighters
  lost.
- Opponent gains (Syreen pods, Mycon regeneration) neither penalise the trainee
  nor reset `low`. Damage that only undoes a gain earns nothing.
- Total credit can't exceed the opponent's starting crew, so this can't be
  farmed.
- A kill always pays the full starting crew less any new lows the trainee didn't
  cause.
- An opponent Syreen above its starting crew must be brought back below it
  before damage earns anything. This is intended: crew is normalised to the
  normal starting crew.

#### Trainee `hp_damage`: follows the crew level

```text
hp_damage = trainee's starting crew - current effective crew   (per life for the Pkunk)
```

- Every loss raises the penalty and every gain lowers it, from any cause and at
  any level, including above the starting crew.
- The total always equals the net change in crew, so this can't be farmed: crew
  can only be regained after a loss that was penalised by the same amount.
- Crew collected above the starting crew stays at risk, so a Syreen has reason
  to protect it.
- Mycon regeneration lowers the penalty directly, giving it a reason to
  regenerate.
- Gains pay only through D_self, so they earn nothing while D_self = 0.
- Placement:
  - Losses caused by the opponent or the environment go on the frame they
    happen.
  - Self-inflicted losses go on the trainee action frame that caused them.
  - Gains go on the frame they happen.
- For the trainee, the damage term depends only on the crew level reached, not
  the path.

- `L`, limpets:
  - The n-th limpet on a ship is worth 0.25 × 0.5^(n−1), so L stays below 0.5.
  - Limpets already on the ship at fight start set n, so a later Vux continues
    the series.
- `sat_damage`: satellite HP removed.

Kill definitions:

- `ship_kills`: 1 per qualifying death, 0.5 for each Pkunk death.
- `sat_kills`: satellites destroyed.

### Checks of the formulas

- **Kill:** a full kill gives exactly K × s for every ship.
  - A Chmmr kill is (1 + 3/3) ÷ 2 = 1. Each satellite is worth 1/6 and the ship
    itself 1/2.
  - The Pkunk's 0.5 per death averages about 0.93 kills per Pkunk.
- **Damage:**
  - Killing all crew gives 0.5 × D × s, or 0.67 × D × s for a Chmmr with all
    satellites destroyed.
  - With limpets at their 0.5 limit, the total is exactly D for every ship. For
    a Chmmr it is (0.5 + 0.5 + 0.5) ÷ 1.5 = 1. Any finite number of limpets
    leaves it just short.
  - This is intended. Limpets, the first especially, cripple a ship, while a
    ship down to 1 crew can still fight at full strength.
- **Vux vs Chmmr example** (D = K = 1.28, s = 1):
  - 3 satellites destroyed;
  - 3 limpets (L = 0.4375);
  - all crew and the kill.

  Damage is (0.5 + 0.4375 + 0.5) ÷ 1.5 = 0.958 D = 1.23, plus the kill 1.28,
  total 2.51. With limpets at their limit it would be 1.0 D = 1.28, total 2.56.
- **Opponent Mycon example** (starting crew 20):

  | Step | Crew | Low-water mark | Trainee credit |
  |---|---|---|---|
  | Trainee hits for 8 | 12 | 12 | +8 |
  | Mycon regenerates 6 | 18 | 12 | 0 |
  | Trainee hits for 10 | 8 | 8 | +4 (only the part below 12) |
  | Planet hit for 3 | 5 | 5 | 0 (not trainee-caused) |
  | Trainee kills it | 0 | 0 | +5 |
  | **Total** | | | **17** |

  - Following the opponent's crew level would also total 17, but would put a
    −6 on whatever the trainee was doing when the Mycon regenerated. The
    new-low rule avoids that.
  - If the Mycon regenerated to full and survived, the trainee would keep its
    +8. The Mycon had to spend battery to regenerate, and the credit is
    bounded.
- **Trainee Syreen example** (start 12, max 42):

  | Step | Crew | Penalty change |
  |---|---|---|
  | Collects 8 | 20 | −8 (credit) |
  | Loses 5 | 15 | +5 |
  | Regains 5 | 20 | −5 (credit) |
  | Collects 2 | 22 | −2 (credit) |
  | Loses 9 | 13 | +9 |
  | **Net** | | **−1, a credit of 1 crew, equal to the net change** |

  - The most a trainee Syreen can be credited is 30 crew above its start:
    0.5 × 30/12 × D_self × s = 1.25 × D_self × s.
  - Crew the song pulls off the opponent is separately opponent damage, paid
    through D_enemy under the new-low rule.

### Attribution

Rewards (opponent harm):

- Only harm the trainee caused counts: the ledger's causal source resolves to
  the trainee. That includes:
  - satellite lasers fired by a trainee Chmmr;
  - marines and fighters launched by the trainee.
- Opponent crew losses not caused by the trainee earn nothing. Examples:
  - planet and asteroid hits;
  - Druuge crew burn;
  - Shofixti self-destruct;
  - fighters or marines lost to the environment.
- Opponent losses the trainee didn't cause still lower the opponent's
  low-water mark. Opponent gains change nothing (see "Opponent `hp_damage`").
- No kill credit for opponent deaths the trainee didn't cause. This matches
  today's factors, which are 0 for planet, Druuge A2 and Shofixti A2 deaths
  (`_enemy_death_reward_credit`).

Penalties (own harm):

- Every own loss counts, whatever the cause, and every own gain counts back
  (see "Trainee `hp_damage`"). Self-inflicted losses count too:
  - Druuge crew burn;
  - Shofixti self-destruct, as both crew damage and a death;
  - launched crew lost for good to enemy fire, asteroids or the planet.
- Notes on launched crew:
  - Kzer-Za fighters route around the planet but can die on asteroids.
  - Orz marines avoid the planet but can hit it under adversarial conditions.
- The harm scheme penalises launched crew only when it is lost for good, not at
  launch. Unlike the classic causal escrow, nothing is charged at launch and
  refunded on recovery.

### UI and storage

- A two-position slider at the top of the Rewards pane switches between Classic
  and Harm. The pane has room for it.
- The pane shows the four harm weights in Harm and the existing sliders in
  Classic.
- Harm weights are stored under new keys. The magnitudes differ from the classic
  per-crew components, so reusing those keys would misread old weights.
- Old configs load as `classic`.
- Training statistics show the harm totals per fight: damage, kill and the two
  penalties.

### Tests

- Damage and kill values for:
  - a 6-crew ship and a 30-cost ship;
  - the Chmmr with satellites;
  - Pkunk damage and deaths, halved, including rebirth;
  - the opponent new-low rule, using the Mycon example: gains ignored, damage
    that only undoes a gain unpaid, losses the trainee didn't cause lowering
    the mark without credit, and a new low partly crossed by one hit;
  - the opponent new-low rule against a ship that can't gain crew, equal to
    plain damage;
  - trainee crew-level following, using the Syreen example: the penalty total
    equals the net crew change, and gains above the start are credited;
  - Pkunk per-life tracking for both the opponent and the trainee;
  - a limpet series continuing from existing limpets.
- Per-frame change rewards sum to the formula totals over a fight.
- Opponent losses the trainee didn't cause earn nothing. Own self-inflicted
  losses are penalised.
- Launched crew:
  - recovered crew costs and earns nothing;
  - trainee-destroyed enemy fighters count;
  - own fighters lost to asteroids are penalised.
- No kill credit for opponent planet, Druuge A2 or Shofixti A2 deaths.
- Causal placement: damage goes on the origin frame, penalties on the effect
  frame.
- Old configs load as classic, and the harm keys round-trip.
- The scheme slider switches the visible weights and persists.

### Expected training impact

- Rewards are sparser and noisier.
- Pair the scheme with a longer half-life (Phase 4) and slower epsilon and
  baseline-use decay.
- Future option, not in scope: a small, decaying imitation term toward baseline
  actions if learning stalls without shaping.

---

## Phase 6: Shared Prioritised Behaviours (last, one ship at a time)

### Target structure

Control priority is decided per control (turn, thrust, A1, A2), as today. Shared
behaviours, each with per-ship parameters and gates:

1. Avoid damage:
   - dodge with a brief turn, then resume the movement goal;
   - shoot down threats;
   - run from threats, as a new option.
2. Avoid the planet, with the Phase 3 horizon.
3. Leave opponent weapon range while recharging, when battery is below the cost
   of the next useful shot.
4. Hold a distance band. This one behaviour covers both "optimal weapon range"
   and "as far as possible", with a preferred side for wrap-around.
5. Keep moving.
6. Fire: from predicted outcomes up close, by rule at long range.
7. Mode switch (Androsynth, Mmrnmrhm), plus a per-ship slot for unique specials:
   Umgah cone, Syreen song, Orz marines, Kzer-Za fighters, shields.

### Process

1. Extract shared behaviours as engine helpers.
2. Add a per-ship parameter table: distance band, recharge threshold, gates and
   planet horizon.
3. Move one ship at a time. Compare crew lost and dealt per minute, still
   frames, and W-L before and after.
4. Commit each ship separately. Revert any ship that regresses.

---

## Decisions

Recorded 2026-11-20.

| # | Topic | Decision |
|---|---|---|
| D1 | Damage normalisation | Keep the 0.5 weight on crew damage. A full crew kill gives 0.5 D, and limpets make up the rest. |
| D2 | Penalty weights | The same formulas as the rewards, with their own weights set separately, as today. Default 0. |
| D3 | Pkunk | Halve both damage and kills: damage given to a Pkunk and damage a Pkunk trainee takes. |
| D4 | Self-inflicted own losses | Penalised. |
| D5 | Opponent deaths the trainee didn't cause | No kill credit, as today. |
| D6 | Half-life range | 1–30 s, default 5 s, with a warning whenever the cutoff exceeds the match frame length set on the Regimen tab. |
| D7 | Satellite HP bar colours | Full white. Then 9 `#00FF00`, 5 `#FFFF00`, 1 `#FF0000`, interpolated between 1 and 9. |
| D8 | Supox end animation | Only on a hit. Other projectiles unchanged. |
| D9 | Reward schemes | Classic and Harm both kept, switched by a slider at the top of the Rewards pane. |
| D10 | Crew gains and losses | Opponent: for every opponent, the trainee is credited only for new lows it causes; gains and losses it didn't cause just move the low-water mark. Trainee: the penalty follows the crew level, so every loss is penalised and every gain credited, netting to the real change in crew. |

---

## Progress Tracker

| Phase | Item | Status | Evidence |
|---|---|---|---|
| 1 | Satellite HP bars | Implemented | `hp_bar_appearance` and `ChmmrSatellite._draw_hp_bar`; `SatelliteHpBarTests` in `tests/test_chmmr.py`; render checked at 10, 9, 7, 5, 3 and 1 HP. Full suite 1,502 tests OK. |
| 1 | Supox end animation on hit only | Implemented | `end_anim_on_hit_only` ability field, set for `SupoxA1`; `destroy_projectile(dealt_damage=...)` is false for laser, area and planet deaths and for pair contacts that dealt no damage; `tests/test_supox_end_animation.py`. |
| 2 | Shared satellite helpers | Not started | |
| 2.1 | Thraddash | Not started | |
| 2.2 | Arilou | Not started | |
| 2.3 | Supox | Not started | |
| 2.4 | Pkunk | Not started | |
| 2.5 | Vux | Not started | |
| 2.6 | Orz | Not started | |
| 2.7 | Kzer-Za (confirm only) | Not started | |
| 2.8 | Androsynth | Not started | |
| 2.9 | Ilwrath | Not started | |
| 2.10 | Utwig | Not started | |
| 2.11 | Earthling | Not started | |
| 2.12 | Umgah | Not started | |
| 2.13 | Slylandro, ZoqFotPik, Spathi, Syreen, Yehat | Not started | |
| 3 | Planet lookahead by agility | Not started | |
| 3A | Pre-emptible held weapons (Chenjesu, Kohr-Ah, Melnorme) | Not started | |
| 4 | Discount half-life setting | Not started | |
| 5 | Harm reward scheme | Not started | |
| 6 | Shared behaviours, per ship | Not started | |
