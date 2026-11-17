# Number Guessing Game

An interactive, modular Python CLI game where players guess a randomly generated number within a set range. Features customizable difficulty levels, input validation, attempt tracking, and replay functionality.

---

## Features

- **Multiple Difficulty Levels**:
  - **Easy**: Numbers from 1 to 50, with 10 attempts.
  - **Medium**: Numbers from 1 to 100, with 7 attempts.
  - **Hard**: Numbers from 1 to 200, with 5 attempts.
- **Robust Input Handling**: Validates user inputs for integer types and allowable range bounds without penalizing remaining attempts on invalid syntax.
- **Directional Feedback**: Informs the player if their guess is "Too high!" or "Too low!".
- **Replayability**: Option to play multiple rounds with different difficulty levels in a single session.
- **Modular Architecture**: Core game logic (`NumberGuessingGame`, `DifficultyConfig`, `GuessResult`) is decoupled from CLI I/O for clean testing and extensibility.

---

## Project Structure

```text
.
├── README.md        # Project documentation
├── game.py          # Core game logic and interactive CLI interface
└── test_game.py     # Comprehensive pytest unit tests
```

---

## Installation & Requirements

- Python 3.8+
- [pytest](https://docs.pytest.org/) for running unit tests

Install dependencies (if needed):
```bash
pip install pytest
```

---

## How to Play

Run the game directly using Python:

```bash
python game.py
```

### Game Flow:
1. Select a difficulty level (1 = Easy, 2 = Medium, 3 = Hard).
2. Enter your guesses when prompted.
3. Receive feedback after each guess (Too high / Too low).
4. Win by guessing the number before running out of attempts!
5. Choose whether to play another round when prompted.

---

## Running Tests

To run the automated test suite with pytest:

```bash
pytest
```

For verbose output:

```bash
pytest -v
```
