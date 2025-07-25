import re
from langchain_core.output_parsers import StrOutputParser


class OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    def extract_answer(
        self,
        text_response: str,
        pattern: str = r"Answer:\s*(.*)"
    ) -> str:
        match_ = re.search(
            pattern,
            text_response,
            re.DOTALL
        )

        if match_:
            answer_text = match_.group(1).strip()
            return answer_text
        else:
            return text_response

        