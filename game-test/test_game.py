"""Unit tests for the Number Guessing Game."""

import pytest
from game import (
    DIFFICULTY_LEVELS,
    DifficultyConfig,
    GuessResult,
    NumberGuessingGame,
    parse_guess,
    prompt_difficulty,
    prompt_replay,
    play_round,
)


class TestDifficultyConfig:
    def test_default_difficulty_levels(self):
        assert "1" in DIFFICULTY_LEVELS
        assert "2" in DIFFICULTY_LEVELS
        assert "3" in DIFFICULTY_LEVELS

        easy = DIFFICULTY_LEVELS["1"]
        assert easy.name == "Easy"
        assert easy.min_val == 1
        assert easy.max_val == 50
        assert easy.max_attempts == 10

        medium = DIFFICULTY_LEVELS["2"]
        assert medium.name == "Medium"
        assert medium.min_val == 1
        assert medium.max_val == 100
        assert medium.max_attempts == 7

        hard = DIFFICULTY_LEVELS["3"]
        assert hard.name == "Hard"
        assert hard.min_val == 1
        assert hard.max_val == 200
        assert hard.max_attempts == 5

    def test_difficulty_description(self):
        diff = DifficultyConfig("Custom", 10, 50, 4)
        assert diff.description() == "Custom (10-50, 4 attempts)"


class TestNumberGuessingGame:
    def test_initialization_with_target(self):
        diff = DifficultyConfig("Test", 1, 10, 3)
        game = NumberGuessingGame(diff, target=5)
        assert game.target == 5
        assert game.max_attempts == 3
        assert game.attempts_taken == 0
        assert game.attempts_remaining == 3
        assert game.won is False
        assert game.is_over is False
        assert game.guesses == []

    def test_initialization_random_target(self):
        diff = DifficultyConfig("Test", 5, 10, 3)
        game = NumberGuessingGame(diff)
        assert 5 <= game.target <= 10

    def test_too_low_guess(self):
        diff = DifficultyConfig("Test", 1, 10, 3)
        game = NumberGuessingGame(diff, target=7)
        result = game.make_guess(3)
        assert result == GuessResult.TOO_LOW
        assert game.attempts_taken == 1
        assert game.attempts_remaining == 2
        assert game.guesses == [3]
        assert not game.is_over
        assert not game.won

    def test_too_high_guess(self):
        diff = DifficultyConfig("Test", 1, 10, 3)
        game = NumberGuessingGame(diff, target=7)
        result = game.make_guess(9)
        assert result == GuessResult.TOO_HIGH
        assert game.attempts_taken == 1
        assert game.attempts_remaining == 2
        assert game.guesses == [9]
        assert not game.is_over
        assert not game.won

    def test_correct_guess(self):
        diff = DifficultyConfig("Test", 1, 10, 3)
        game = NumberGuessingGame(diff, target=7)
        result = game.make_guess(7)
        assert result == GuessResult.CORRECT
        assert game.attempts_taken == 1
        assert game.attempts_remaining == 2
        assert game.is_over is True
        assert game.won is True

    def test_exhaust_attempts_loss(self):
        diff = DifficultyConfig("Test", 1, 10, 2)
        game = NumberGuessingGame(diff, target=5)
        res1 = game.make_guess(1)
        assert res1 == GuessResult.TOO_LOW
        assert not game.is_over
        assert game.attempts_remaining == 1

        res2 = game.make_guess(2)
        assert res2 == GuessResult.TOO_LOW
        assert game.attempts_remaining == 0
        assert game.is_over is True
        assert game.won is False

    def test_out_of_bounds_guess(self):
        diff = DifficultyConfig("Test", 1, 10, 3)
        game = NumberGuessingGame(diff, target=5)
        with pytest.raises(ValueError, match="between 1 and 10"):
            game.make_guess(0)
        with pytest.raises(ValueError, match="between 1 and 10"):
            game.make_guess(11)
        # Verify attempts are not consumed on invalid bounds
        assert game.attempts_taken == 0
        assert game.attempts_remaining == 3

    def test_guess_after_game_over(self):
        diff = DifficultyConfig("Test", 1, 10, 1)
        game = NumberGuessingGame(diff, target=5)
        game.make_guess(5)
        assert game.is_over is True
        with pytest.raises(RuntimeError, match="game is already over"):
            game.make_guess(5)


class TestInputParsing:
    def test_parse_guess_valid(self):
        assert parse_guess("42", 1, 100) == 42
        assert parse_guess("  15 \n", 1, 100) == 15

    def test_parse_guess_non_integer(self):
        with pytest.raises(ValueError, match="not a valid integer"):
            parse_guess("abc", 1, 100)
        with pytest.raises(ValueError, match="not a valid integer"):
            parse_guess("12.5", 1, 100)
        with pytest.raises(ValueError, match="not a valid integer"):
            parse_guess("", 1, 100)

    def test_parse_guess_out_of_range(self):
        with pytest.raises(ValueError, match="between 1 and 50"):
            parse_guess("0", 1, 50)
        with pytest.raises(ValueError, match="between 1 and 50"):
            parse_guess("51", 1, 50)


class TestInteractivePrompts:
    def test_prompt_difficulty_valid(self):
        responses = iter(["2"])
        messages = []
        result = prompt_difficulty(
            input_func=lambda _: next(responses),
            print_func=messages.append,
        )
        assert result == DIFFICULTY_LEVELS["2"]

    def test_prompt_difficulty_retry_on_invalid(self):
        responses = iter(["invalid", "5", "1"])
        messages = []
        result = prompt_difficulty(
            input_func=lambda _: next(responses),
            print_func=messages.append,
        )
        assert result == DIFFICULTY_LEVELS["1"]
        assert any("Invalid choice" in m for m in messages)

    @pytest.mark.parametrize("val", ["y", "Y", "yes", "YES", " Yes "])
    def test_prompt_replay_yes(self, val: str):
        result = prompt_replay(input_func=lambda _: val, print_func=lambda _: None)
        assert result is True

    @pytest.mark.parametrize("val", ["n", "N", "no", "NO", " No "])
    def test_prompt_replay_no(self, val: str):
        result = prompt_replay(input_func=lambda _: val, print_func=lambda _: None)
        assert result is False

    def test_prompt_replay_retry_invalid(self):
        responses = iter(["maybe", "123", "y"])
        messages = []
        result = prompt_replay(
            input_func=lambda _: next(responses),
            print_func=messages.append,
        )
        assert result is True
        assert any("Please enter 'y' for yes or 'n' for no." in m for m in messages)


class TestPlayRound:
    def test_play_round_win(self):
        diff = DifficultyConfig("Test", 1, 10, 3)
        inputs = iter(["5", "7"])  # 5 is low, 7 is correct
        prints = []
        game = play_round(
            diff,
            target=7,
            input_func=lambda _: next(inputs),
            print_func=prints.append,
        )
        assert game.won is True
        assert game.attempts_taken == 2
        assert any("Congratulations" in p for p in prints)
        assert any("Too low!" in p for p in prints)

    def test_play_round_loss(self):
        diff = DifficultyConfig("Test", 1, 10, 2)
        inputs = iter(["10", "9"])  # both too high, exhausts 2 attempts
        prints = []
        game = play_round(
            diff,
            target=5,
            input_func=lambda _: next(inputs),
            print_func=prints.append,
        )
        assert game.won is False
        assert game.is_over is True
        assert any("Game Over!" in p for p in prints)

    def test_play_round_handles_invalid_input(self):
        diff = DifficultyConfig("Test", 1, 10, 2)
        inputs = iter(["invalid", "99", "5"])  # invalid string, out of bounds, correct
        prints = []
        game = play_round(
            diff,
            target=5,
            input_func=lambda _: next(inputs),
            print_func=prints.append,
        )
        assert game.won is True
        assert game.attempts_taken == 1
        assert any("Invalid input:" in p for p in prints)
