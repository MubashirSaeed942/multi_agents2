import json
from crewai.flow.flow import Flow, listen, start
from new_agent.crew import UserGuider
from  pydantic import BaseModel
from typing import Optional

class LLm_response(BaseModel):
    prompt: str = ""
    response : str = ""

class WorkflowState(BaseModel):
    user_input :str = ""
    llm_response :str = ""



class Workflow(Flow[LLm_response]):
    state: type[WorkflowState] = WorkflowState


    
    @start()
    def user_input(self) -> str:
        print("input is giving by user")
        self.state.user_input = input("Ask Me! : ")
        return self.state.user_input
        
    
    @listen(user_input)
    def llm_call(self):
        result = UserGuider().crew().kickoff(inputs= {"question" : self.state.user_input})
        self.state.llm_response = result.raw
        return self.state.llm_response
    
    @listen(llm_call)
    def save_file(self):
        response_data = {
            "prompt" : self.state.user_input,
            "response" : self.state.llm_response

        }
        filename  = "llm_response.json"
        with open(filename , "w", encoding= "utf-8") as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)

        print(f"Response saved to {filename}")


def kickflow():
    object = Workflow()
    object.kickoff()        