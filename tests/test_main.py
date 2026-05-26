import unittest

from letscode.main import ChatBubble, ChatEntry


class ChatBubbleTest(unittest.TestCase):
    def test_chat_bubble_uses_role_classes(self) -> None:
        bubble = ChatBubble(ChatEntry("user", "How should I start?"))
        child = list(bubble.compose())[0]

        self.assertTrue(bubble.has_class("chat-row"))
        self.assertTrue(bubble.has_class("user"))
        self.assertTrue(child.has_class("chat-bubble"))
        self.assertTrue(child.has_class("user"))

    def test_chat_bubble_preserves_agent_role(self) -> None:
        bubble = ChatBubble(ChatEntry("agent", "Look for a hash map invariant."))
        child = list(bubble.compose())[0]

        self.assertTrue(bubble.has_class("agent"))
        self.assertTrue(child.has_class("agent"))


if __name__ == "__main__":
    unittest.main()
