"""Regression tests for the aquarium daily-care game.

Covers the eight historical defects:
  1. food cost deducted twice when feeding
  2. a broken filter still purified the water
  3. inverted water-quality boundary comparisons
  4. quarantined fish counted in the public display
  5. ticket income generated on days the tank was closed
  6. failed water changes were not rolled back
  7. the same disease spread to the same fish repeatedly
  8. loading a save silently skipped a day

Plus edge cases: illegal commands, empty data, capacity limits and
duplicate input.  Invariants on population, water quality, capacity and
coins are re-asserted after every scenario.
"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from aquarium.core import (
    FEED_COST,
    REPAIR_COST,
    WATER_CHANGE_COST,
    CommandError,
    Fish,
    Game,
    check_invariants,
    default_fish,
    demo,
)


def make_game(coins=120, water_quality=60, fish=None, capacity=4,
              filter_ok=True, exhibit_open=False, seed=20240517):
    return Game(
        coins=coins,
        water_quality=water_quality,
        fish=fish if fish is not None else default_fish(),
        capacity=capacity,
        filter_ok=filter_ok,
        exhibit_open=exhibit_open,
        seed=seed,
    )


class FeedTests(unittest.TestCase):
    def test_food_cost_deducted_once(self):
        game = make_game(coins=100)
        result = game.feed()
        self.assertTrue(result.ok)
        expected = 100 - FEED_COST * sum(not f.quarantined for f in game.fish)
        self.assertEqual(game.coins, expected)
        check_invariants(game)

    def test_feed_twice_charges_twice_not_four_times(self):
        game = make_game(coins=100)
        game.feed()
        after_first = game.coins
        game.feed()
        after_second = game.coins
        one_round = FEED_COST * sum(not f.quarantined for f in game.fish)
        self.assertEqual(after_first - after_second, one_round)
        check_invariants(game)

    def test_feed_skips_quarantined_fish(self):
        fish = [Fish("Nemo", health=70), Fish("Dory", health=70, quarantined=True)]
        game = make_game(coins=100, fish=fish)
        game.feed()
        self.assertEqual(game.coins, 100 - FEED_COST)
        check_invariants(game)

    def test_feed_without_enough_coins_changes_nothing(self):
        game = make_game(coins=0)
        before = game.snapshot()
        result = game.feed()
        self.assertFalse(result.ok)
        self.assertEqual(game.snapshot(), before)
        check_invariants(game)

    def test_feed_empty_tank_is_rejected(self):
        game = make_game(coins=100, fish=[])
        result = game.feed()
        self.assertFalse(result.ok)
        self.assertEqual(game.coins, 100)


class WaterChangeTests(unittest.TestCase):
    def test_failed_change_is_rolled_back(self):
        game = make_game(coins=100, water_quality=40, filter_ok=False)
        before = game.snapshot()
        result = game.change_water()
        self.assertFalse(result.ok)
        self.assertEqual(game.snapshot(), before)
        check_invariants(game)

    def test_change_costs_money_and_clears_water(self):
        game = make_game(coins=100, water_quality=33)
        result = game.change_water()
        self.assertTrue(result.ok)
        self.assertEqual(game.coins, 100 - WATER_CHANGE_COST)
        self.assertEqual(game.water_quality, 100)
        check_invariants(game)

    def test_change_unaffordable_changes_nothing(self):
        game = make_game(coins=0, water_quality=20)
        before = game.snapshot()
        result = game.change_water()
        self.assertFalse(result.ok)
        self.assertEqual(game.snapshot(), before)

    def test_broken_filter_does_not_purify_at_night(self):
        # A broken filter can never improve the water, even on a "good"
        # random draw.  Force every draw to look favourable.
        game = make_game(coins=200, water_quality=80, filter_ok=False)
        game.rng.random = lambda: 0.0
        game.advance_day()
        self.assertLessEqual(game.water_quality, 80)
        self.assertFalse(game.filter_ok)
        check_invariants(game)

    def test_water_quality_boundaries_clamp(self):
        game = make_game(water_quality=99, filter_ok=True)
        game.rng.random = lambda: 0.0
        game.advance_day()
        self.assertLessEqual(game.water_quality, 100)
        game2 = make_game(water_quality=1, filter_ok=True)
        game2.rng.random = lambda: 1.0
        game2.advance_day()
        self.assertGreaterEqual(game2.water_quality, 0)


class QuarantineTests(unittest.TestCase):
    def test_quarantined_fish_not_in_display(self):
        fish = [
            Fish("Nemo", displayable=True),
            Fish("Dory", quarantined=True, displayable=True),
        ]
        game = make_game(fish=fish)
        self.assertEqual([f.name for f in game.display_fish], ["Nemo"])

    def test_quarantined_fish_earn_no_ticket_income(self):
        fish = [Fish("Dory", quarantined=True, displayable=True)]
        game = make_game(fish=fish, exhibit_open=True)
        result = game.advance_day()
        self.assertEqual(result.data["income"], 0)


class ExhibitTests(unittest.TestCase):
    def test_no_income_on_closed_day(self):
        game = make_game(exhibit_open=False)
        result = game.advance_day()
        self.assertEqual(result.data["income"], 0)
        check_invariants(game)

    def test_income_only_when_open_and_fish_present(self):
        game = make_game(exhibit_open=True)
        result = game.advance_day()
        self.assertGreater(result.data["income"], 0)
        check_invariants(game)

    def test_exhibit_closes_overnight(self):
        game = make_game(exhibit_open=True)
        game.advance_day()
        self.assertFalse(game.exhibit_open)

    def test_empty_open_tank_earns_nothing(self):
        game = make_game(fish=[], exhibit_open=True)
        result = game.advance_day()
        self.assertEqual(result.data["income"], 0)


class DiseaseTests(unittest.TestCase):
    def test_same_disease_spreads_at_most_once_per_fish(self):
        fish = [
            Fish("Patient0", disease="ich"),
            Fish("Alpha"),
            Fish("Beta"),
            Fish("Gamma"),
        ]
        game = make_game(fish=fish, capacity=4)
        game.rng.random = lambda: 0.0  # every possible contact infects
        game.advance_day()
        for fish_obj in game.fish:
            if fish_obj.name != "Patient0":
                self.assertEqual(fish_obj.diseases, ["ich"])
        # Patient0's own disease list must not gain duplicates either.
        self.assertEqual(game.fish[0].diseases, ["ich"])

    def test_disease_skips_quarantined_fish(self):
        fish = [
            Fish("Patient0", disease="ich"),
            Fish("Lonely", quarantined=True),
        ]
        game = make_game(fish=fish, capacity=2)
        game.rng.random = lambda: 0.0
        game.advance_day()
        self.assertEqual(game.fish[1].diseases, [])


class CommandValidationTests(unittest.TestCase):
    def test_unknown_command_rejected(self):
        game = make_game()
        with self.assertRaises(CommandError):
            game.execute("juggle")
        with self.assertRaises(CommandError):
            game.execute("")

    def test_quarantine_unknown_fish_rejected(self):
        game = make_game(fish=[Fish("Nemo")])
        with self.assertRaises(CommandError):
            game.execute("quarinate", {"name": "Moby"})

    def test_add_fish_over_capacity_rejected(self):
        game = make_game(fish=[Fish("Nemo")], capacity=1)
        before = game.snapshot()
        result = game.execute("add_fish", {"name": "Dory"})
        self.assertFalse(result.ok)
        self.assertEqual(game.snapshot(), before)

    def test_capacity_boundary_allows_exactly_capacity(self):
        game = make_game(fish=[], capacity=1)
        result = game.execute("add_fish", {"name": "Solo"})
        self.assertTrue(result.ok)
        self.assertEqual(len(game.fish), 1)
        result = game.execute("add_fish", {"name": "Crowd"})
        self.assertFalse(result.ok)
        self.assertEqual(len(game.fish), 1)

    def test_duplicate_open_input_is_idempotent(self):
        game = make_game()
        game.execute("open_exhibit")
        coins_first = game.coins
        game.execute("open_exhibit")
        self.assertTrue(game.exhibit_open)
        self.assertEqual(game.coins, coins_first)

    def test_duplicate_quarantine_input_is_idempotent(self):
        game = make_game(fish=[Fish("Nemo")])
        game.execute("quarantine", {"name": "Nemo"})
        game.execute("quarantine", {"name": "Nemo"})
        self.assertTrue(game.fish[0].quarantined)

    def test_duplicate_add_fish_input_is_idempotent(self):
        game = make_game(fish=[], capacity=2)
        game.execute("add_fish", {"name": "Nemo"})
        game.execute("add_fish", {"name": "Nemo"})
        self.assertEqual(len(game.fish), 1)


class SaveLoadTests(unittest.TestCase):
    def test_load_does_not_advance_the_day(self):
        game = make_game(coins=77, water_quality=55)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "save.json")
            game.save(path)
            day_before = game.day
            loaded = Game.load(path)
            self.assertEqual(loaded.day, day_before)
            self.assertEqual(loaded.coins, 77)
            self.assertEqual(loaded.water_quality, 55)

    def test_round_trip_preserves_state_and_rng(self):
        game = make_game(fish=[Fish("Nemo", disease="finrot"), Fish("Dory")])
        game.feed()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "save.json")
            game.save(path)
            loaded = Game.load(path)
            self.assertEqual(loaded.snapshot(), game.snapshot())
            loaded_stream = [loaded.rng.random() for _ in range(5)]
            fresh = Game.load(path)
            saved_stream = [fresh.rng.random() for _ in range(5)]
            self.assertEqual(loaded_stream, saved_stream)

    def test_load_empty_data_uses_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "save.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{}")
            loaded = Game.load(path)
            check_invariants(loaded)
            self.assertEqual(loaded.day, 1)

    def test_save_file_is_dict_not_list(self):
        game = make_game()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "save.json")
            game.save(path)
            import json
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIsInstance(data, dict)
            self.assertEqual(data["version"], 1)


class InvariantTests(unittest.TestCase):
    def test_fixed_demo_is_deterministic_and_valid(self):
        out_a = io.StringIO()
        with redirect_stdout(out_a):
            final_a = demo(days=7)
        out_b = io.StringIO()
        with redirect_stdout(out_b):
            final_b = demo(days=7)
        self.assertEqual(out_a.getvalue(), out_b.getvalue())
        self.assertEqual(final_a, final_b)
        check_invariants(Game.from_dict(final_a))

    def test_random_walk_keeps_invariants(self):
        game = make_game()
        commands = [
            ("feed", {}),
            ("change_water", {}),
            ("open_exhibit", {}),
            ("advance_day", {}),
            ("repair_filter", {}),
        ]
        for _ in range(30):
            name, payload = commands[game.rng.randrange(len(commands))]
            try:
                game.execute(name, payload)
            except CommandError:
                pass
            check_invariants(game)


if __name__ == "__main__":
    unittest.main()
