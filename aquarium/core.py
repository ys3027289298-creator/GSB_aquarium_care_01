"""Core state machine for the aquarium daily-care game.

A single :class:`Game` owns the whole simulation.  Commands mutate the
state during one in-game day; :meth:`Game.advance_day` resolves the
nightly events (tickets, disease spread, filter wear, water quality and
fish health) and rolls the day over.

All randomness flows through one seeded ``random.Random`` instance whose
state is persisted with the save, so fixed-seed demos are reproducible
even across save/load boundaries.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SAVE_VERSION = 1
DEFAULT_SEED = 20240517

FEED_COST = 3
WATER_CHANGE_COST = 15
REPAIR_COST = 40
TICKET_PRICE = 2

MIN_VISITORS = 4
MAX_VISITORS = 12
FILTER_BREAK_CHANCE = 0.15
SPREAD_CHANCE = 0.5
FILTER_PURIFY = 8
DIRT_ADD = 6
GOOD_WATER = 50
GOOD_WATER_HEALTH = 3
BAD_WATER_HEALTH = -4
FED_HEALTH = 4
MAX_HEALTH = 100
MIN_HEALTH = 0


class AquariumError(Exception):
    """Base class for aquarium errors."""


class CommandError(AquariumError):
    """Raised when a command name or its arguments are invalid."""


@dataclass(init=False)
class Fish:
    name: str
    health: int = 60
    quarantined: bool = False
    displayable: bool = True
    diseases: List[str] = field(default_factory=list)

    def __init__(
        self,
        name: str,
        health: int = 60,
        quarantined: bool = False,
        displayable: bool = True,
        diseases: Optional[List[str]] = None,
        disease: Optional[str] = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise CommandError("fish name must be a non-empty string")
        self.name = name
        self.health = max(MIN_HEALTH, min(MAX_HEALTH, int(health)))
        self.quarantined = bool(quarantined)
        self.displayable = bool(displayable)
        if diseases:
            self.diseases = list(diseases)
        elif disease:
            self.diseases = [disease]
        else:
            self.diseases = []

    @property
    def disease(self) -> Optional[str]:
        return self.diseases[0] if self.diseases else None

    @disease.setter
    def disease(self, value: Optional[str]) -> None:
        if value:
            self.diseases = [value]
        else:
            self.diseases = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "health": self.health,
            "quarantined": self.quarantined,
            "displayable": self.displayable,
            "diseases": list(self.diseases),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fish":
        diseases = list(data.get("diseases") or [])
        if not diseases and data.get("disease"):
            diseases = [data["disease"]]
        return cls(
            name=str(data["name"]),
            health=int(data.get("health", 60)),
            quarantined=bool(data.get("quarantined", False)),
            displayable=bool(data.get("displayable", True)),
            diseases=diseases,
        )


@dataclass
class Result:
    """Outcome of executing a command."""

    ok: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


def default_fish() -> List[Fish]:
    return [
        Fish("Bubbles", health=72),
        Fish("Gill", health=65),
        Fish("Coral", health=58),
    ]


class Game:
    def __init__(
        self,
        coins: int = 120,
        water_quality: int = 60,
        fish: Optional[List[Fish]] = None,
        capacity: int = 4,
        filter_ok: bool = True,
        exhibit_open: bool = False,
        day: int = 1,
        seed: int = DEFAULT_SEED,
        rng_state: Optional[Any] = None,
    ) -> None:
        self.coins = int(coins)
        self.water_quality = max(0, min(100, int(water_quality)))
        self.fish: List[Fish] = list(fish) if fish is not None else default_fish()
        self.capacity = int(capacity)
        self.filter_ok = bool(filter_ok)
        self.exhibit_open = bool(exhibit_open)
        self.day = int(day)
        self.seed = seed
        self.rng = random.Random(seed)
        if rng_state is not None:
            self.rng.setstate(rng_state)
        check_invariants(self)

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    @property
    def display_fish(self) -> List[Fish]:
        """Healthy, non-quarantined fish the public can see."""
        return [
            fish_obj
            for fish_obj in self.fish
            if fish_obj.displayable and not fish_obj.quarantined
        ]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "coins": self.coins,
            "water_quality": self.water_quality,
            "capacity": self.capacity,
            "filter_ok": self.filter_ok,
            "exhibit_open": self.exhibit_open,
            "fish": [fish_obj.to_dict() for fish_obj in self.fish],
        }

    # ------------------------------------------------------------------
    # Daytime commands
    # ------------------------------------------------------------------
    def feed(self) -> Result:
        eaters = [fish_obj for fish_obj in self.fish if not fish_obj.quarantined]
        if not eaters:
            return Result(False, "no fish to feed")
        cost = FEED_COST * len(eaters)
        if self.coins < cost:
            return Result(False, f"need {cost} coins to feed, have {self.coins}")
        self.coins -= cost
        for fish_obj in eaters:
            fish_obj.health = min(MAX_HEALTH, fish_obj.health + FED_HEALTH)
        return Result(True, f"fed {len(eaters)} fish for {cost} coins",
                      {"cost": cost, "fed": len(eaters)})

    def change_water(self) -> Result:
        if self.coins < WATER_CHANGE_COST:
            return Result(
                False,
                f"need {WATER_CHANGE_COST} coins, have {self.coins}",
            )
        if not self.filter_ok:
            return Result(False, "filter is broken; water change refused")
        self.coins -= WATER_CHANGE_COST
        self.water_quality = 100
        return Result(True, "water changed", {"cost": WATER_CHANGE_COST})

    def quarantine(self, name: str) -> Result:
        fish_obj = self._find_fish(name)
        if fish_obj.quarantined:
            return Result(True, f"{name} is already in quarantine")
        fish_obj.quarantined = True
        return Result(True, f"{name} moved to quarantine")

    def release(self, name: str) -> Result:
        fish_obj = self._find_fish(name)
        if not fish_obj.quarantined:
            return Result(True, f"{name} is not in quarantine")
        fish_obj.quarantined = False
        return Result(True, f"{name} returned to the tank")

    def open_exhibit(self) -> Result:
        if self.exhibit_open:
            return Result(True, "exhibit is already open")
        self.exhibit_open = True
        return Result(True, "exhibit opened to visitors")

    def close_exhibit(self) -> Result:
        self.exhibit_open = False
        return Result(True, "exhibit closed")

    def add_fish(self, name: str, health: int = 60) -> Result:
        if any(fish_obj.name == name for fish_obj in self.fish):
            return Result(True, f"{name} is already in the tank")
        if len(self.fish) >= self.capacity:
            return Result(False, f"tank is at capacity ({self.capacity})")
        self.fish.append(Fish(name=name, health=health))
        return Result(True, f"{name} added")

    def repair_filter(self) -> Result:
        if self.filter_ok:
            return Result(True, "filter is already working")
        if self.coins < REPAIR_COST:
            return Result(
                False,
                f"need {REPAIR_COST} coins to repair, have {self.coins}",
            )
        self.coins -= REPAIR_COST
        self.filter_ok = True
        return Result(True, "filter repaired", {"cost": REPAIR_COST})

    # ------------------------------------------------------------------
    # Nightly resolution
    # ------------------------------------------------------------------
    def advance_day(self) -> Result:
        open_today = self.exhibit_open
        shown = list(self.display_fish) if open_today else []
        visitors = self.rng.randint(MIN_VISITORS, MAX_VISITORS) if shown else 0
        income = visitors * TICKET_PRICE * len(shown)
        self.coins += income

        spread_events = self._spread_diseases()

        # Water quality: a working filter may purify, dirt always adds.
        draw = self.rng.random()
        if self.filter_ok and draw < 0.8:
            self.water_quality -= DIRT_ADD
            self.water_quality += FILTER_PURIFY
        else:
            self.water_quality -= DIRT_ADD
        self.water_quality = max(0, min(100, self.water_quality))

        # Filter wears out independently of the purification draw.
        if self.filter_ok and self.rng.random() < FILTER_BREAK_CHANCE:
            self.filter_ok = False

        # Fish health follows the (correct) water-quality boundary.
        if self.water_quality >= GOOD_WATER:
            health_delta = GOOD_WATER_HEALTH
        else:
            health_delta = BAD_WATER_HEALTH
        for fish_obj in self.fish:
            fish_obj.health = max(
                MIN_HEALTH,
                min(MAX_HEALTH, fish_obj.health + health_delta),
            )

        self.exhibit_open = False
        self.day += 1
        check_invariants(self)
        return Result(
            True,
            f"day {self.day - 1} resolved",
            {
                "income": income,
                "visitors": visitors,
                "shown": len(shown),
                "water_quality": self.water_quality,
                "filter_ok": self.filter_ok,
                "spread": spread_events,
            },
        )

    def _spread_diseases(self) -> List[Dict[str, str]]:
        """Resolve one nightly round of disease spread.

        Contagious fish are snapshotted up front so a fish that catches a
        disease tonight cannot spread it the same night, and each target
        is touched at most once per disease.  Quarantined fish neither
        spread nor catch anything.
        """
        events: List[Dict[str, str]] = []
        sources = [
            fish_obj
            for fish_obj in self.fish
            if fish_obj.diseases and not fish_obj.quarantined
        ]
        targets = [
            fish_obj
            for fish_obj in self.fish
            if not fish_obj.quarantined
        ]
        for source in sources:
            for disease in list(source.diseases):
                for target in targets:
                    if target is source or disease in target.diseases:
                        continue
                    if self.rng.random() < SPREAD_CHANCE:
                        target.diseases.append(disease)
                        events.append(
                            {"disease": disease, "to": target.name}
                        )
        return events

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = self.snapshot()
        data["version"] = SAVE_VERSION
        data["seed"] = self.seed
        data["rng_state"] = list(self.rng.getstate()[1])
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Game":
        data = data or {}
        fish_data = data.get("fish")
        if fish_data is None:
            fish: Optional[List[Fish]] = default_fish()
        else:
            fish = [Fish.from_dict(item) for item in fish_data]
        seed = data.get("seed", DEFAULT_SEED)
        rng_state = None
        if data.get("rng_state"):
            rng_state = (3, tuple(data["rng_state"]), None)
        return cls(
            coins=int(data.get("coins", 120)),
            water_quality=int(data.get("water_quality", 60)),
            fish=fish,
            capacity=int(data.get("capacity", 4)),
            filter_ok=bool(data.get("filter_ok", True)),
            exhibit_open=bool(data.get("exhibit_open", False)),
            day=int(data.get("day", 1)),
            seed=seed,
            rng_state=rng_state,
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "Game":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise AquariumError("save file must contain a JSON object")
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------
    def _find_fish(self, name: str) -> Fish:
        if not isinstance(name, str) or not name.strip():
            raise CommandError("fish name must be a non-empty string")
        for fish_obj in self.fish:
            if fish_obj.name == name:
                return fish_obj
        raise CommandError(f"no fish named {name!r}")

    def execute(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Result:
        payload = payload or {}
        if not isinstance(command, str) or not command.strip():
            raise CommandError("command must be a non-empty string")
        if not isinstance(payload, dict):
            raise CommandError("payload must be an object")

        if command == "feed":
            return self.feed()
        if command == "change_water":
            return self.change_water()
        if command == "open_exhibit":
            return self.open_exhibit()
        if command == "close_exhibit":
            return self.close_exhibit()
        if command == "advance_day":
            return self.advance_day()
        if command == "repair_filter":
            return self.repair_filter()
        if command == "quarantine":
            return self.quarantine(payload.get("name", ""))
        if command == "release":
            return self.release(payload.get("name", ""))
        if command == "add_fish":
            return self.add_fish(
                payload.get("name", ""),
                int(payload.get("health", 60)),
            )
        raise CommandError(f"unknown command: {command!r}")


def check_invariants(game: Game) -> None:
    """Assert population, water, capacity and coin invariants."""
    if game.coins < 0:
        raise AquariumError(f"negative coins: {game.coins}")
    if not 0 <= game.water_quality <= 100:
        raise AquariumError(
            f"water quality out of range: {game.water_quality}"
        )
    if len(game.fish) > game.capacity:
        raise AquariumError(
            f"population {len(game.fish)} exceeds capacity {game.capacity}"
        )
    if game.capacity < 0:
        raise AquariumError(f"negative capacity: {game.capacity}")
    if game.day < 1:
        raise AquariumError(f"invalid day: {game.day}")
    names = [fish_obj.name for fish_obj in game.fish]
    if len(names) != len(set(names)):
        raise AquariumError("duplicate fish names")
    for fish_obj in game.fish:
        if not 0 <= fish_obj.health <= 100:
            raise AquariumError(
                f"{fish_obj.name} health out of range: {fish_obj.health}"
            )
        if len(fish_obj.diseases) != len(set(fish_obj.diseases)):
            raise AquariumError(f"{fish_obj.name} has duplicate diseases")


def demo(days: int = 7, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Run a fixed-seed demonstration schedule and print a daily report."""
    game = Game(seed=seed)
    print(f"=== Aquarium demo (seed={seed}, {days} days) ===")
    print(
        f"day {game.day}: start | coins={game.coins} "
        f"water={game.water_quality} fish={len(game.fish)}"
    )
    schedule = [
        ("feed", {}),
        ("repair_filter", {}),
        ("change_water", {}),
        ("open_exhibit", {}),
    ]
    for _ in range(days):
        for command, payload in schedule:
            result = game.execute(command, payload)
            print(f"  {command}: {'ok' if result.ok else 'skip'} - {result.message}")
        result = game.advance_day()
        print(
            f"day {game.day}: coins={game.coins} water={game.water_quality} "
            f"filter={'ok' if result.data['filter_ok'] else 'broken'} "
            f"income={result.data['income']} "
            f"shown={result.data['shown']}"
        )
    print("=== final ===")
    print(json.dumps(game.snapshot(), indent=2, sort_keys=True))
    check_invariants(game)
    return game.snapshot()
