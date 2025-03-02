from crewai import Agent, Crew, Process, Task , LLM
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
load_dotenv()

llm = LLM(model="gemini/gemini-1.5-flash", temperature=0)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(GEMINI_API_KEY)

text_file = TextFileKnowledgeSource(
	file_paths=["myfile.txt"]
)
@CrewBase
class UserGuider():
	"""UserGuider Crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def user_guider(self) -> Agent:
		return Agent(
			config=self.agents_config['user_guider'], 
			verbose=True,
			knowledge_sources=[text_file],
			llm=llm,
            embedder={
               "provider": "google",
               "config": {
                  "model": "models/text-embedding-004",
                  "api_key": GEMINI_API_KEY,
                         }
            }
		)

	@agent
	def user_suggestor(self) -> Agent:
		return Agent(
			config=self.agents_config['user_suggestor'],
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def guide_task(self) -> Task:
		return Task(
			config=self.tasks_config['guide_task'],
		)

	@task
	def suggest_task(self) -> Task:
		return Task(
			config=self.tasks_config['suggest_task'],
			context=[self.guide_task()],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the UserGuider crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			 agents=[self.user_guider(), self.user_suggestor()],  # Properly initialize agents
             tasks=[self.guide_task(), self.suggest_task()],  # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			knowledge_sources=[text_file],
		    embedder={
               "provider": "google",
               "config": {
                  "model": "models/text-embedding-004",
                  "api_key": GEMINI_API_KEY,
        }
    }
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

