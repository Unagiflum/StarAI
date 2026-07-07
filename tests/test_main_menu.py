import os
import unittest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src import main


class MainMenuItemsTests(unittest.TestCase):
    def test_train_ai_is_hidden_when_training_is_unavailable(self):
        labels = [label for label, _ in main.main_menu_items(training_available=False)]

        self.assertNotIn("Train AI", labels)
        self.assertIn("Play Game", labels)

    def test_train_ai_is_shown_when_training_is_available(self):
        labels = [label for label, _ in main.main_menu_items(training_available=True)]

        self.assertIn("Train AI", labels)


if __name__ == "__main__":
    unittest.main()
