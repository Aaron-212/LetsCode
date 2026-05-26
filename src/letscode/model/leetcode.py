from pydantic import BaseModel, Field, RootModel


class LeetCodeQuestion(BaseModel):
    class CodeSnippet(BaseModel):
        language: str
        code: str

    questionId: int
    title: str
    difficulty: str
    content: str  # Rich HTML
    hints: list[str]  # Rich HTML


class LeetCodeQuestionList(RootModel):
    class Question(BaseModel):
        questionId: int = Field(validation_alias="frontend_id")
        title: str
        title_slug: str
        difficulty: str
        url: str  # LeetCode URL
        paid_only: bool
        has_solution: bool

    root: list[Question]
