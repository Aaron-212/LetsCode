import unittest

from langchain_core.messages import HumanMessage

from letscode.agent import LetsCodeAgent, LetsCodeAgentState, tool_factory


class AgentToolFactoryTest(unittest.TestCase):
    def test_read_user_solution_tool_returns_solution_context(self) -> None:
        state = LetsCodeAgentState(
            current_solution_code="class Solution:\n    pass\n"
        )
        tools = {tool.name: tool for tool in tool_factory(state)}

        self.assertIn("read_user_solution", tools)
        self.assertEqual(
            tools["read_user_solution"].invoke({}),
            "class Solution:\n    pass\n",
        )

    def test_current_solution_tool_remains_available(self) -> None:
        state = LetsCodeAgentState(
            current_solution_code="class Solution:\n    pass\n"
        )
        tools = {tool.name: tool for tool in tool_factory(state)}

        self.assertEqual(
            tools["get_current_solution_code"].invoke({}),
            "class Solution:\n    pass\n",
        )


class AgentStateTest(unittest.TestCase):
    def test_clear_chat_history_removes_messages(self) -> None:
        state = LetsCodeAgentState(messages=[HumanMessage(content="hello")])
        agent = LetsCodeAgent.__new__(LetsCodeAgent)
        agent.state = state

        agent.clear_chat_history()
        self.assertEqual(state.messages, [])


if __name__ == "__main__":
    unittest.main()
