"""Python Number Guessing Game.

Features:
- Multiple difficulty levels (Easy, Medium, Hard)
- Configurable guess bounds and attempt limits
- State tracking for guesses and win/loss conditions
- Interactive CLI with replay capability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
import sys
from typing import Callable, List, Optional


class GuessResult(Enum):
    """Result of evaluating a player's guess."""
    CORRECT = "correct"
    TOO_HIGH = "too_high"
    TOO_LOW = "too_low"


@dataclass(frozen=True)
class DifficultyConfig:
    """Configuration for a game difficulty level."""
    name: str
    min_val: int
    max_val: int
    max_attempts: int

    def description(self) -> str:
        return f"{self.name} ({self.min_val}-{self.max_val}, {self.max_attempts} attempts)"


DIFFICULTY_LEVELS: dict[str, DifficultyConfig] = {
    "1": DifficultyConfig("Easy", 1, 50, 10),
    "2": DifficultyConfig("Medium", 1, 100, 7),
    "3": DifficultyConfig("Hard", 1, 200, 5),
}


class NumberGuessingGame:
    """Encapsulates the state and logic for a single round of the game."""

    def __init__(self, difficulty: DifficultyConfig, target: Optional[int] = None) -> None:
        self.difficulty = difficulty
        self.target = (
            target
            if target is not None
            else random.randint(difficulty.min_val, difficulty.max_val)
        )
        self.max_attempts = difficulty.max_attempts
        self.attempts_taken = 0
        self.guesses: List[int] = []
        self.won = False
        self.is_over = False

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.max_attempts - self.attempts_taken)

    def validate_guess(self, guess: int) -> None:
        """Validate whether a guess is within allowable range."""
        if guess < self.difficulty.min_val or guess > self.difficulty.max_val:
            raise ValueError(
                f"Guess must be between {self.difficulty.min_val} and {self.difficulty.max_val}."
            )

    def make_guess(self, guess: int) -> GuessResult:
        """Process a guess and return the result."""
        if self.is_over:
            raise RuntimeError("The game is already over.")

        self.validate_guess(guess)

        self.attempts_taken += 1
        self.guesses.append(guess)

        if guess == self.target:
            self.won = True
            self.is_over = True
            return GuessResult.CORRECT

        if self.attempts_remaining == 0:
            self.is_over = True

        if guess > self.target:
            return GuessResult.TOO_HIGH
        else:
            return GuessResult.TOO_LOW


def parse_guess(raw_input: str, min_val: int, max_val: int) -> int:
    """Parse and validate user guess input string."""
    cleaned = raw_input.strip()
    try:
        val = int(cleaned)
    except ValueError:
        raise ValueError(f"'{raw_input}' is not a valid integer.")

    if val < min_val or val > max_val:
        raise ValueError(f"Guess must be between {min_val} and {max_val}.")
    return val


def prompt_difficulty(
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> DifficultyConfig:
    """Prompt the user to select a difficulty level."""
    print_func("\nSelect Difficulty Level:")
    for key, diff in DIFFICULTY_LEVELS.items():
        print_func(f"  [{key}] {diff.description()}")

    while True:
        choice = input_func("Enter choice (1-3): ").strip()
        if choice in DIFFICULTY_LEVELS:
            return DIFFICULTY_LEVELS[choice]
        print_func("Invalid choice. Please choose 1, 2, or 3.")


def prompt_replay(
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> bool:
    """Prompt the user if they want to play another game."""
    while True:
        choice = input_func("\nWould you like to play again? (y/n): ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print_func("Please enter 'y' for yes or 'n' for no.")


def play_round(
    difficulty: DifficultyConfig,
    target: Optional[int] = None,
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> NumberGuessingGame:
    """Play a single round of the game interactively."""
    game = NumberGuessingGame(difficulty, target=target)
    print_func(f"\n--- Starting {difficulty.name} Mode ---")
    print_func(
        f"I'm thinking of a number between {difficulty.min_val} and {difficulty.max_val}."
    )
    print_func(f"You have {game.max_attempts} attempts. Good luck!\n")

    while not game.is_over:
        prompt_text = f"Attempt {game.attempts_taken + 1}/{game.max_attempts} - Enter your guess: "
        user_input = input_func(prompt_text)
        try:
            guess = parse_guess(user_input, difficulty.min_val, difficulty.max_val)
        except ValueError as err:
            print_func(f"Invalid input: {err}")
            continue

        result = game.make_guess(guess)
        if result == GuessResult.CORRECT:
            print_func(
                f"Congratulations! You guessed the number {game.target} in {game.attempts_taken} attempts!"
            )
        elif result == GuessResult.TOO_HIGH:
            print_func("Too high!")
        elif result == GuessResult.TOO_LOW:
            print_func("Too low!")

        if not game.won:
            if game.is_over:
                print_func(f"Game Over! The number was {game.target}.")
            else:
                print_func(f"Attempts remaining: {game.attempts_remaining}\n")

    return game


def main() -> None:
    """Main CLI entrypoint for the number guessing game."""
    print("=" * 40)
    print("   Welcome to the Number Guessing Game!   ")
    print("=" * 40)

    try:
        while True:
            difficulty = prompt_difficulty()
            play_round(difficulty)
            if not prompt_replay():
                print("\nThanks for playing! Goodbye!")
                break
    except (KeyboardInterrupt, EOFError):
        print("\n\nGame aborted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
