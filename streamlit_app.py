import os
import sys
import re

import streamlit as st

from crewai import Crew,Task,Agent,Process
from textwrap import dedent
import requests

#from content_creator_agents import ContentCreationAgents

from streamlit_quill import st_quill

import time

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_anthropic.llms import AnthropicLLM

from crewai_tools import SerperDevTool
from crewaitools.search_tools import SearchTools
from crewaitools.browser_tools import BrowserTools
from crewai.tasks.task_output import TaskOutput

from langchain.tools import tool
from bs4 import BeautifulSoup

# from content_creator_agents import ContentCreationAgents as ContentAgents
# from content_creator_tasks import content_creation_tasks as ContentTasks
# from content_creator_crew import content_creation_tasks
#from content_creator_sidebar import contentsidebar as sb

from dotenv import load_dotenv
load_dotenv()

mycontent = ""
part = ""
scene = ""
beat = ""

class ContentTools:
    @tool("Read webpage content")
    def read_content(url: str) -> str:
        """Read content from a webpage."""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = soup.get_text()
        return text_content[:5000]

def streamlit_callback(step_output):
    # This function will be called after each step of the agent's execution
    st.markdown("---")
    for step in step_output:
        if isinstance(step, tuple) and len(step) == 2:
            action, observation = step
            if isinstance(action, dict) and "tool" in action and "tool_input" in action and "log" in action:
                st.markdown(f"# Action")
                st.markdown(f"**Tool:** {action['tool']}")
                st.markdown(f"**Tool Input** {action['tool_input']}")
                st.markdown(f"**Log:** {action['log']}")
                st.markdown(f"**Action:** {action['Action']}")
                st.markdown(
                    f"**Action Input:** ```json\n{action['tool_input']}\n```")
            elif isinstance(action, str):
                st.markdown(f"**Action:** {action}")
            else:
                st.markdown(f"**Action:** {str(action)}")

            st.markdown(f"**Observation**")
            if isinstance(observation, str):
                observation_lines = observation.split('\n')
                for line in observation_lines:
                    if line.startswith('Title: '):
                        st.markdown(f"**Title:** {line[7:]}")
                    elif line.startswith('Link: '):
                        st.markdown(f"**Link:** {line[6:]}")
                    elif line.startswith('Snippet: '):
                        st.markdown(f"**Snippet:** {line[9:]}")
                    elif line.startswith('-'):
                        st.markdown(line)
                    else:
                        st.markdown(line)
            else:
                st.markdown(str(observation))
        else:
            st.markdown(step)

#display the console processing on streamlit UI
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Check if the data contains 'task' information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(self.colors)  # Increment color index and wrap around if necessary

            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        if "Lead Market Analyst" in cleaned_data:
            # Apply different color 
            cleaned_data = cleaned_data.replace("Lead Market Analyst", f":{self.colors[self.color_index]}[Lead Market Analyst]")
        if "Creative Content Creator" in cleaned_data:
            cleaned_data = cleaned_data.replace("Creative Content Creator", f":{self.colors[self.color_index]}[Creative Content Creator]")
        if "Chief Content Strategist" in cleaned_data:
            cleaned_data = cleaned_data.replace("Chief Content Strategist", f":{self.colors[self.color_index]}[Chief Content Strategist]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

st.set_page_config(page_title="Content Creator Studio powered by CrewAI", layout="wide")

# Initialize session state variables
if 'result_analysis' not in st.session_state:
    st.session_state['result_analysis'] = None
if 'result_strategy' not in st.session_state:
    st.session_state['result_strategy'] = None
if 'result_outline' not in st.session_state:
    st.session_state['result_outline'] = None

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    #st.sidebar.image("./img/ContentCreationStudioIcon.png",width=150)
    #st.header("Content Creation Studio - Refactored")

    #selected_key = st.text_input("API Key", type="password")
    st.subheader("Select Content Type")
    content_type = st.selectbox(label="Select Type",options=["Operating Model","Playbook","eBook","Article","Blog Post","Social Media Post","Essay","Report","PowerPoint streamPresentation","Novel","Learning Course","Instruction Manual"])
    st.divider()
    LLM_selection = st.selectbox(label="Select Model:", options=["Claude Haiku-S","Claude Sonnet-M","Claude Opus-L","Mixtral","Mistral 7B", "Google Gemini"])
    model_temp = st.number_input("Temperature", step=0.1, max_value=1.0, value=0.3)
    with st.sidebar.expander(label="API Keys",expanded=False):
        openai_user_api_key = st.text_input(label = ":key: OpenAI API Key:",type="password",placeholder="Enter model API Key",key="openai_user_api_key")
        azure_user_api_key = st.text_input(label = ":key: Azure OpenAI API Key:",type="password",placeholder="Enter model API Key",key="azure_user_api_key")
        mistral_user_api_key = st.text_input(label = ":key: Mistral API Key:",type="password",placeholder="Enter model API Key",key="mistral_user_api_key")
        gemini_user_api_key = st.text_input(label = ":key: Google Gemini API Key:",type="password",placeholder="Enter model API Key",key="gemini_user_api_key")
        anthropic_user_api_key = st.text_input(label = ":key: Anthropic Claude API Key:",type="password",placeholder="Enter model API Key",key="anthropic_user_api_key")
        cohere_user_api_key = st.text_input(label = ":key: Cohere API Key:",type="password",placeholder="Enter model API Key",key="cohere_user_api_key")
    st.session_state["model_temp"] = model_temp
    st.divider()
    st.subheader("Content Outline Depth")
    num_parts = st.number_input("Parts",min_value=1,max_value=10,value=5,step=1,key="num_parts")
    num_sections = st.number_input("Sections per Part", min_value=1, max_value=5, value=3, step=1, key="num_sections")
    num_subsections = st.number_input("Subsections per Section", min_value=1, max_value=5, value=3, step=1, key="num_subsections")
    Wordcount = st.slider("Wordcount per Subsection", 250,750,(500,600),step=50)
    with st.expander(label="Content Depth Summary"):
        st.write(f"{num_parts} Parts")
        st.write(f"{num_sections} sections per part")
        st.write(f"{num_subsections} subsections per section")
        st.write(f"{Wordcount} words per subsection")
        st.write(f"Total Estimated Wordcount: {num_parts * num_sections * num_subsections * Wordcount[0]} - {num_parts * num_sections * num_subsections * Wordcount[1]}")
    st.session_state["wordcount_target"] = Wordcount
    st.divider()
    st.subheader("Advanced Settings")
    with st.expander(label="Writing Style Settings"):
        content_depth = st.selectbox(label="Content Depth",help="Deductive reasoning helps you solve problems by using what you already know is true.",index=3,options=("Best Suited for Content","Elementary (Grade 1-6)", "Middle School (Grade 7-9)", "High School (Grade 10-12)", "Undergraduate", "Graduate (Bachelor Degree)", "Master's", "Doctoral Candidate (Ph.D Candidate)", "Postdoc", "Ph.D"),key="depth")
        learning_style = st.selectbox(label="Learning Style",options=("Best Suited for Content","Active", "Global","Intuitive", "Reflective","Verbal","Visual"),key="learningstyle")
        comm_style = st.selectbox(label="Communication Style",options=("Best Suited for Content","Formal", "Layman", "Socratic", "Story Telling", "Textbook"),key="commstyle")
        tone_style = st.selectbox(label="Tone Style",options=("Best Suited for Content","Aspirational","Conversational","Encouraging","Firm","Friendly","Humorous","Informative","Inspirational","Lighthearted","Neutral","Persuasive","Spartan"),key="tone")
        reasoning = st.selectbox(label="Reasoning Framework",options=("Best Suited for Content","Abductive", "Analogical", "Causal","Deductive", "Inductive"),key="reasoning")
    st.divider()

    st.markdown(
        "For more information about CrewAI, see [here](https://github.com/joaomdmoura/crewAI)."
    )
    mistral_key=os.getenv('mistral_key')
    Mixtral_llm = ChatMistralAI(mistral_api_key=mistral_key, model="open-mixtral-8x7b",temperature=model_temp,verbose=True)
    Mistral7B_llm = ChatMistralAI(mistral_api_key=mistral_key, model="open-mistral-7b",temperature=model_temp,verbose=True)

    GoogleGemini_llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=model_temp, google_api_key=os.getenv("GOOGLE_API_KEY"),verbose=True)

    ClaudeHaiku_llm = ChatAnthropic(model_name='claude-3-haiku-20240307',temperature=model_temp,verbose=True)
    ClaudeSonnet_llm = ChatAnthropic(model_name='claude-3-sonnet-20240229',temperature=model_temp,verbose=True)
    ClaudOpus_llm = ChatAnthropic(model_name='claude-3-opus-20240229',temperature=model_temp,verbose=True)

    if LLM_selection == "Mixtral":
        llm=Mixtral_llm
    elif LLM_selection == "Mistral 7B":
        llm=Mistral7B_llm
    elif LLM_selection == "Google Gemini":
        llm=GoogleGemini_llm
    elif LLM_selection == "Claude Haiku-S":
        llm=ClaudeHaiku_llm
    elif LLM_selection == "Claude Sonnet-M":
        llm=ClaudeSonnet_llm
    elif LLM_selection == "Claude Opus-L":
        llm=ClaudOpus_llm

if 'content_strategies' not in st.session_state:
    st.session_state['content_strategies'] = [None for _ in range(num_parts)] * 5

if 'content_outlines' not in st.session_state:
    st.session_state['content_outlines'] = [None for _ in range(num_parts)] * 5

if 'subsection_content' not in st.session_state:
    st.session_state['subsection_content'] = [None for _ in range(num_subsections)] * (num_parts * num_sections * num_subsections)

#Crew Agents
class ContentCreationAgents():
  def __init__(self,mycontent):
      self.mycontent = mycontent
  
  def lead_market_analyst(self,mycontent):
    return Agent(
        role='Lead Market Analyst',
        goal=f"""Prepare in-depth market insights on content trends, pain points, opportunities to help address pain points, audience, and market saturation of 
        content to aid in creating targeted {content_type} content based on the following content idea: {mycontent}.""",
        backstory=
        f"""With a talent for uncovering hidden market trends and a deep understanding of content analysis, you have guided numerous projects to successfully cater
         to their ideal audience. Your analytical skills are unmatched, making you a cornerstone in the strategy development process.""",
        tools=[
                SearchTools.search_internet,
                BrowserTools.scrape_and_summarize_website,
                ContentTools().read_content
                ],
        llm=llm,
        memory=False,
        max_rpm=10,
        verbose=True,
        step_callback=streamlit_callback
        )
  def chief_content_strategist(self,mycontent):
    return Agent(
        role='Chief Content Strategist',
        goal=dedent(f"""
        Develop a content strategy for {content_type} content. Utilize the market analysis conducted by the market analyst to guide the creation of {content_type} content. 
        Incorporate relevant information about logical and progressive sequencing, concepts of chapter beats, scene beats, outlines, and writing engaging content. Create {num_parts} Parts, each consisting of {num_sections} sections, and each section containing {num_subsections} subsections. 
        Ensure that each section includes key, actionable, and insightful takeaways. Additionally, include a provocative introduction, a conclusion, and a call to action, sparking introspective conversation or action among the audience.
        """),
        backstory=f"""A visionary in content planning and strategy, Parts, sections, chapters, any type of content. You have mastery of concepts like instruction manuals, logical sequencing of content, chapter beats, creating content in the most engaging tone of the audience, brilliant and engaging style, you excel at transforming market insights of readers into actionable content plans. 
        Your strategies are known for capturing audience interests and driving engagement.
        When you strategize content, you aim to not just impart information but also move people to think differently and take action. Your passion lies in using your gift of storytelling and incorporating the following guidelines:
        1. Develop ideas for a Provocative Question or Statement for the Creative content creator to use: Start the {content_type} sections with a question or statement that immediately grabs attention. This could revolve around a common challenge or a controversial opinion in the field of agile metrics and analytics. The purpose is to provoke thought and stir the reader's curiosity from the very first sentence. 
        2. Incorporate a Anecdote where helpful: Include a brief, engaging story that demonstrates the firsthand experience with the transformative power of agile metrics and analytics. This story should be relatable and highlight a specific instance where agile metrics significantly impacted an IT project's outcome. The narrative should showcase the author's expertise and the real-world application of the principles discussed in the {content_type}. 
        3. Outline the Promise of the content in the introduction only: Clearly articulate what the reader will gain by reading this {content_type} in the introduction. This section should highlight the unique insights, methodologies, or strategies the reader will learn and how this knowledge will empower them in their career or projects within the agile framework. It's important to make a compelling value proposition that promises practical benefits and newfound understanding. 
        4. End each part with an Engaging Question or Interactive Element: Conclude the introduction with a rhetorical question or a prompt that encourages the reader to think actively about their own experiences or challenges with agile metrics. This could also be an invitation to engage with the {content_type}'s content in a way that feels personal and directly relevant to their own work or interests.
        """,
        llm=llm,
        memory=True,
        max_rpm=10,
        verbose=True,
        step_callback=streamlit_callback,
    )
  
  def creative_content_creator(self,mycontent):
    return Agent(
        role='Creative Content Creator',
        goal=dedent(f"""
        Write engaging and informative {content_type} content for each part, section, and subsection in the provided sequence and outline by the Chief Content Strategist.
        Your {content_type} content should be coherent, engaging, and aim to encourage critical thinking among readers about their work environments and practices.
        Each {content_type} Part should be expanded into coherent sub-section content, following a logical sequence and including any additional considerations provided by the Chief Content Strategist.
        You use Metacognitive Knowledge techniques to create opportunities for reader to reflect on and monitor their strengths and areas of improvement, and plan how to overcome current difficulties. 
        Providing enough challenge for readers to develop effective strategies, but not so difficult that they struggle to apply a strategy. 
        Encourage readers to think about their own thinking and learning processes. The goal is to make readers are aware of how they learn, understand the strategies that work best for them, and encourage them to take control of their learning. 
        When developing content, your goal is to not only impart information but also move people to think differently and take action. You should use your storytelling skills and incorporate the following guidelines:
        1. Begin each section with a Provocative Question or Statement: Start each section with a question or statement that immediately grabs attention, revolving around a common challenge or a controversial opinion in the field of agile metrics and analytics, to provoke thought and stir the reader's curiosity from the very first sentence.
        2. Incorporate an Anecdote where helpful: Include a brief, engaging story that demonstrates firsthand experience with the transformative power of agile metrics and analytics. This story should be relatable and highlight a specific instance where agile metrics significantly impacted an IT project's outcome, showcasing expertise and the real-world application of the principles discussed in the {content_type}.
        3. Outline the Promise of the content in the introduction only: Clearly articulate what the reader will gain by reading the content in the introduction. Highlight the unique insights, methodologies, or strategies the reader will learn and how this knowledge will empower them in their career or projects within the agile framework, making a compelling value proposition that promises practical benefits and newfound understanding.
        4. End each part with an Engaging Question or Interactive Element: Conclude each introduction with a rhetorical question or a prompt that encourages the reader to think actively about their own experiences or challenges with agile metrics. This could also be an invitation to engage with the {content_type}'s content in a way that feels personal and directly relevant to their own work or interests.
        """),
        backstory="""A storyteller at heart, you blend creativity with insight to produce content that educates, entertains, and enlightens readers. Your written words have the power to captivate, inspire, and convey complex principles.

        """,
        llm=llm,
        memory=True,
        max_rpm=10,
        verbose=True,
        step_callback=streamlit_callback,
        #callbacks=[CustomStreamlitCallbackHandler(color="white")]
        )

agents = ContentCreationAgents(mycontent)

#Crew Tasks
class ContentCreation_tasks():
    def market_analysis_task(self,agent):
        return Task(
            description=dedent(f"""Summarize provided content, search the internet for similar content by topic and theme, assess the market saturation and relevance, and identify the best audience for the {content_type}."""),
            expected_output="A comprehensive report including analysis of similar content, assessment of market saturation, relevance, and recommendations for target audience demographics and preferences.",
            agent=agent,
            #callback=callback_function
            #tools=[search_tool]
        )
    def content_strategy_task(self, agent,part_idx):
        return Task(
            description=dedent(f"""Develop a content strategy based on this idea: \n\n {mycontent} \n
            Incorporate the following market analysis and insights:\n
            {st.session_state['result_analysis']} \n
            Provide the content creator instructions, and suggestions on topics, tone, scene beats and sequence of content to make the content as compelling easy to understand, and applicable as possible for 
            the target audience provided from the market analysis. Do not include word count estimate.
            """),
            expected_output=dedent(
            f"""
            Part {part_idx+1} content strategy only. A detailed content strategy for only part {part_idx+1} of the content, guiding the creation of content. Strategy should include the tone, scene beats, topic, subtopics, key points, pain points to address for each topic.
            """),
            agent=agent,
            context=[self.market_analysis_task(agent)],
            #callback=callback_function
        )
    def content_creation_task_0(self, agent):
       return Task(
            description=dedent(f"""Based on the content idea and content strategy, take a step back and use your expert content sequencing skills to structure and write the content outline in {num_parts} distinct parts based on details provided by the Chief Content Strategist ensuring you are incorporating the content guidelines.
            The content idea that needs to be addressed is as as follows: {mycontent} \n
            Content strategist strategy is as follows: \n
            {content_strategy}
            """),
            expected_output=dedent(f"""Outline only, based on the content strategy from the chief content strategist You will create {num_parts} Parts, each part has {num_sections} section, each section has {num_subsections} Subsections, and at the end section are key takeaways for each section. 
            The outline should follow the sequence and any additional considerations provided by the Chief Content Strategist in the content strategy.
            Remember the llm will only all 3000 output tokens for the response so make your outline concise, and focused yet explanatory enough, so the content creator has a clear roadmap to follow.
            """),
            agent=agent,
            #context=[self.content_strategy_task(agent)],
            #callback=callback_function
        )
    def outline_creation_task(self, agent):
       return Task(
            description=dedent(f"""Based on the content idea and content strategy, take a step back and use your expert content sequencing skills to structure and write the content outline for only Part {part_idx} based on details provided by the Chief Content Strategist ensuring you are incorporating the content guidelines.
            The content idea that needs to be addressed is as as follows: {mycontent} \n
            Content strategist strategy is as follows: \n
            {content_strategy}
            """),
            expected_output=dedent(f"""You will create Part {part_idx} Outline only, based on the content strategy. Part {part_idx} must have {num_sections} sections, each section has {num_subsections} Subsections, and at the last section are key takeaways for each section. 
            The outline should follow the sequence and any additional considerations provided by the Chief Content Strategist in the content strategy.
            Remember the llm will only all 3000 output tokens for the response so make your outline concise, and focused yet explanatory enough, so the content creator has a clear roadmap to follow.
            """),
            agent=agent,
            #context=[self.content_strategy_task(agent)],
            #callback=callback_function
        )
    def create_content_task(self,agent, part, scene, beat,outline):
        return Task(
            description=f"""Carefully craft this {content_type} to it's intended audience. Write at a {content_depth} level, for a reader with a {learning_style} learning style, and {comm_style} communication style, in {tone_style} tone, using {reasoning} framework.
            Incorporate Metacognitive Knowledge content such as Reflective Activities, Metacognitive Prompts.
            Write part {part} scene beat {scene} sub-sceen beat {beat} content written according to the instructions by the content strategist.
            The content should follow the outline sequence and any additional considerations provided by the Chief Content Strategist in the content strategy and outline documents.
            Do not number the paragraphs. Each paragraph should have a meaningful heading. Ensure the content flows logically and coherently, following outline from the content creator Outline:       
            \n{outline}\n
            """,
            expected_output=f"""{content_type} Content for part {part} scene beat {scene} sub-sceen beat {beat} based on the outline of the content (no introduction) completely 
            written in at least between {Wordcount[0]} and {Wordcount[1]} words, incorporating the instructions and inights from the content strategist, and guidelines and 
            outline of the content creator. Do not number the paragraphs. Each section should have a heading. The content must flow logically and coherently and follow 
            the outline. engage readers and encourage them to think critically about their own work environments and practices. This could involve questions posed directly 
            to the reader, exercises at the end of chapters, or scenarios that invite readers to consider what they would do.""", 
            agent=agent
            )
    def refinestrategytask(self,agent, content_strategy):
        return Task(
            description=f"""
            Refine content strategy based on the provided information: \n{refine_strategy_input}\n
            Below is the existing Content strategy, analyze the existing content strategy and make refinements using it as context:
            {content_strategy}
            """,
            expected_output="An updated content strategy, with only requested refinements, the other unspecified parts of the strategy remain unchanged.",
            context=[self.content_creation_task_0(agent)],
            agent=agent,
            #callback=callback_function
        )
    def RefineOutlineTask(self,agent,content_outline, refine_outline_input):
        return Task(
            description=f"""
            Refine content outline based on the provided information:\n{refine_outline_input}\n
            Below is the existing Content outline, analyze the existing content outline and make refinements using it as context:
            {content_outline}
            """,
            expected_output="An updated content outline, with only requested refinements, the other unspecified parts of the outline remain unchanged.",
            context=[self.content_creation_task_0(agent)],
            agent=agent,
            #callback=callback_function
            )

class OperatingModelAgents():
    def __init__(self,mycontent):
      self.mycontent = mycontent

    def strategy_consultant_agent(self,mycontent):
        return Agent(
            role='Strategy Consultant Agent',
            goal=dedent(f"""
            Construct a detailed Operating Model framework that outlines the essential components and mechanisms through which businesses can achieve strategic alignment and operational excellence. 
            This model should serve as a blueprint for consultants to assist their clients in streamlining operations, enhancing efficiency, and fostering innovation.As an expert in your field, start 
            by drafting the introduction to Operating Models, emphasizing their significance for strategic objectives.
            """),
            backstory=dedent(f"""
            With a rich background in consulting, you've mastered the art of bridging the gap between strategic goals and operational realities across diverse industries.
            Your insights come from years of experience, witnessing the transformational impact of effectively implemented Operating Models.
            This deep understanding fuels your mission to guide the creation of dynamic Operating Models that not only align with strategic objectives but also enhance organizational agility and efficiency.
            Tasked with drafting the introduction to Operating Models, you're poised to share your knowledge, aiming to empower consultants with the tools needed for driving strategic alignment and operational excellence in their clients' organizations.
            """),
            llm=llm,
            memory=True,
            max_rpm=10,
            verbose=True,
            step_callback=streamlit_callback,
            #callbacks=[CustomStreamlitCallbackHandler(color="white")]
        )
    
    def business_process_analyst_agent():
        return Agent(
            role='Business Process Analyst Agent',
            goal=dedent(f"""
            To meticulously analyze and refine business processes, ensuring they deliver maximum value to customers while operating at peak efficiency. This involves deep dives into existing workflows, 
            identifying bottlenecks and inefficiencies, and recommending process improvements that enhance both the quality and speed of service delivery.
            """),
            backstory="""Sharpened by years in the trenches of business process optimization, your analytical prowess and meticulous attention to detail have carved paths through the complexities of various sectors. 
            Your mission: to refine operations, eliminate excess, and drive forward innovation. At the core of your philosophy is a simple truth — the strength of an organization's processes dictates its success. 
            Efficiency and effectiveness in these processes directly translate to customer value. As the Business Process Analyst Agent, you are charged with the critical examination and improvement of the Operating Model's business processes. 
            Your seasoned expertise guides you to craft solutions that boost operational performance and, ultimately, enhance customer satisfaction.
            """,
            llm=llm,
            memory=True,
            max_rpm=10,
            verbose=True,
            step_callback=streamlit_callback,
            #callbacks=[CustomStreamlitCallbackHandler(color="white")]
        )
    def organizational_design_agent():
        return Agent(
            role='Organizational Design Specialist Agent',
            goal=dedent(f"""
            Design an organizational structure that clearly defines roles, responsibilities, and reporting lines, aligning them directly with business goals to enhance efficiency and effectiveness.
            """),
            backstory="""Your path through organizational design has been marked by a focus on essential elements: structure, clarity, and alignment. With experience across sectors, you've streamlined organizations to operate more effectively, 
            always with an eye on the ultimate objective. As the Organizational Design Specialist Agent, your task is to refine the organizational blueprint, ensuring each component is precisely aligned with the organization's strategic aims, ready to deliver optimal performance.
            """,
            llm=llm,
            memory=True,
            max_rpm=10,
            verbose=True,
            step_callback=streamlit_callback,
            #callbacks=[CustomStreamlitCallbackHandler(color="white")]
        )
    def technology_systems_architect_agent():
        return Agent(
            role='Technology Systems Architect Agent',
            goal=dedent(f"""
            Architect and refine technology systems to seamlessly support and enhance operational execution, ensuring digital infrastructure is both robust and agile.
            """),
            backstory="""With a pragmatic eye on the digital landscape, you've engineered technology systems that not only withstand the pressures of modern operations but also propel organizations forward.
            Your expertise lies in creating infrastructures that are as efficient as they are innovative, always aligned with the operational demands.
            As the Technology Systems Architect Agent, your mission is to construct and optimize technological frameworks that enable seamless operational execution, underpinning the organization's strategic endeavors with cutting-edge digital support.
            """,
            llm=llm,
            memory=True,
            max_rpm=10,
            verbose=True,
            step_callback=streamlit_callback,
            #callbacks=[CustomStreamlitCallbackHandler(color="white")]
        )
    def information_flow_coordinator_agent():
        return Agent(
            role='Information Flow Coordinator Agent',
            goal=dedent(f"""
            Streamline the organization's information flow, designing mechanisms for efficient data and knowledge management that enhance decision-making and operational agility.
            """),
            backstory="""Your expertise in managing information flow has transformed the way organizations handle data and knowledge, making complex systems accessible and efficient. 
            With a focus on clarity and utility, you've crafted solutions that ensure the right information reaches the right places at the right time. 
            As the Information Flow Coordinator Agent, your objective is to implement information management systems that support seamless operations and informed decision-making across the organization.
            """,
            llm=llm,
            memory=True,
            max_rpm=10,
            verbose=True,
            step_callback=streamlit_callback,
            #callbacks=[CustomStreamlitCallbackHandler(color="white")]
        )
    def governance_culture_advisor_agent():
        return Agent(
            role='Governance and Culture Advisor Agent',
            goal=dedent(f"""
            Advise on and implement governance frameworks and decision-making processes that align with operational goals, while fostering a culture and leadership style conducive to those objectives.
            """),
            backstory="""Through your advisory roles, you've shaped governance structures and cultural norms that align with and advance operational goals. Your approach balances rigor with flexibility, 
            ensuring frameworks support decision-making and operational agility. As the Governance and Culture Advisor Agent, you aim to integrate governance and cultural practices that promote strategic alignment and organizational cohesion.
            """,
            llm=llm,
            memory=True,
            max_rpm=10,
            verbose=True,
            step_callback=streamlit_callback,
            #callbacks=[CustomStreamlitCallbackHandler(color="white")]
        )
    def implementation_manager_agent():
        return Agent(
            role='Implementation Manager Agent',
            goal=dedent(f"""
            Guide the detailed, step-by-step implementation of the Operating Model, incorporating effective risk and change management strategies to ensure smooth adoption and operational integration.
            """),
            backstory="""Your track record in managing complex implementations has made you adept at navigating transitions and mitigating risks. You've mastered the art of guiding organizations through change, 
            ensuring each step is clear, strategic, and aligned with long-term goals. As the Implementation Manager Agent, your focus is on orchestrating the adoption of the Operating Model with precision, ensuring risks are managed and changes are seamlessly integrated.
            """,
            llm=llm,
            memory=True,
            max_rpm=10,
            verbose=True,
            step_callback=streamlit_callback,
            #callbacks=[CustomStreamlitCallbackHandler(color="white")]
        )
    def implementation_manager_agent():
        return Agent(
            role='Case Study Curator Agent',
            goal=dedent(f"""
            Collect and compile transformative case studies of Operating Model implementations, highlighting best practices and key lessons learned to inform and guide future strategies.
            """),
            backstory="""Your experience lies in distilling complex transformations into insightful case studies, identifying patterns of success and areas for improvement. 
            You have a keen eye for the impactful and the instructional, making you adept at curating content that educates and inspires. 
            As the Case Study Curator Agent, you focus on gathering and presenting case studies that exemplify operational excellence and strategic alignment.
            """,
            llm=llm,
            memory=True,
            max_rpm=10,
            verbose=True,
            step_callback=streamlit_callback,
            #callbacks=[CustomStreamlitCallbackHandler(color="white")]
        )
    def outline_curation_agent(self,mycontent):
        return Agent(
            role='Outline Curator Agent',
            goal=dedent(f"""
            Provide only the part of the outline that is being requested with all of it's sections and subsections.
            Here is the outline: \n
            Operating Model Framework Outline
            Part 1. Introduction
            Section 1.1. Definition of an Operating Model
            Section 1.2. Importance of Aligning Strategy with Operations
            Section 1.3. Overview of the Framework Structure
            Part 2. Business Processes
            Section 2.1. Identification of Critical Business Processes
            Section 2.2. Analysis of Current Process Efficiencies
            Section 2.3. Recommendations for Process Optimization
            Part 3. Organizational Structure
            Section 3.1. Design Principles for Organizational Structure
            Section 3.2. Roles, Responsibilities, and Reporting Lines
            Section 3.3. Alignment with Business Processes and Goals
            Part 4. Technology Systems
            Section 4.1. Overview of Required Technology Infrastructure
            Section 4.2. Applications and Tools Supporting Operations
            Section 4.3. Integration with Business Processes and Organizational Structure
            Part 5. Information Flow Mechanisms
            Section 5.1. Data and Knowledge Management Strategies
            Section 5.2. Designing Effective Information Flow within the Organization
            Section 5.3. Supporting Decision-Making and Operations
            Part 6. Governance and Culture
            Section 6.1. Governance Frameworks and Decision-Making Processes
            Section 6.2. Cultivating a Supportive Organizational Culture
            Section 6.3. Leadership Styles and Their Impact on Operational Goals
            Part 7. Implementation Roadmap
            Section 7.1. Steps for Adopting the Operating Model
            Section 7.2. Incorporating Risk Management and Change Management Strategies
            Section 7.3. Timeline and Milestones for Implementation
            Part 8. Case Studies
            Section 8.1. Overview of Selected Case Studies
            Section 8.2. Analysis of Best Practices and Lessons Learned
            Section 8.3. Application of the Operating Model in Different Industries
            Part 9. Conclusion
            Section 9.1. Recap of the Operating Model Framework
            Section 9.2. The Value of Strategic Alignment and Operational Excellence
            Section 9.3. Next Steps for Consultants and Organizations
            Part 10. Appendices
            Section 10.1. Glossary of Terms
            Section 10.2. Additional Resources and Readings
            Section 10.3. Templates and Tools for Implementation
            """),
            backstory="""You are very precise and respond only with the requested part of the outline, without any additional context or commentary. Your response will be a concise and well-structured outline of the specified part, including all its sections and subsections.
            """,
            llm=llm,
            memory=True,
            max_rpm=10,
            verbose=True,
            step_callback=streamlit_callback,
        )

class OperatingModelTasks():
    def OutlineCurator_Task(self,agent,part_idx,mycontent):
        return Task(
            description=f"""
            Provide only the outline for only Part {part_idx+1} of the Operating Model. \n
            Context: \n {mycontent}
            """,
            expected_output=dedent(f"""
            only Part {part_idx+1} of the Operating Model outline.
            """),
            context=[],
            agent=agent,
            #callback=callback_function
            )
    def OutlineCreator_Task(self,agent,part_idx,mycontent):
        return Task(
            description=f"""
            Create only the outline for only Part {part_idx+1} of the Operating Model. Ask Outline Curator Agent for the part needed.\n
            Context: \n {mycontent}
            """,
            expected_output=dedent(f"""
            A comprehensive outline for only Part {part_idx+1} of the Operating Model framework, outlining its importance, purpose, and the benefits of strategic alignment.
            """),
            context=[self.OutlineCurator_Task()],
            agent=agent,
            #callback=callback_function
            )
    def IntroductionDrafting_Task(self,agent):
        return Task(
            description=f"""
            Draft an introduction that defines an Operating Model specific to the context given, explaining its importance in aligning business strategy with operational execution. Highlight how this framework acts as a foundation for achieving operational excellence. \n
            Context: \n {mycontent}
            """,
            expected_output=dedent(f"""
            A comprehensive introduction section for the Operating Model framework, outlining its importance, purpose, and the benefits of strategic alignment.
            """),
            context=[],
            agent=agent,
            #callback=callback_function
            )
    def BusinessProcessElaboration_Task(self,agent):
        return Task(
            description=f"""
            Analyze and detail the critical business processes necessary for delivering customer value. Identify inefficiencies and recommend optimizations to improve efficiency and effectiveness.
            """,
            expected_output=dedent(f"""
            A detailed analysis of critical business processes, including recommendations for optimization and efficiency improvements.
            """),
            context=[],
            agent=agent,
            #callback=callback_function
            )
    def OrganizationalStructureDesign_Task(self,agent):
        return Task(
            description=f"""
            Design an organizational structure that supports the operational execution and aligns with business goals. This includes defining roles, responsibilities, and reporting lines based on the detailed business processes.
            """,
            expected_output=dedent(f"""
            A clear and coherent organizational structure that aligns roles and responsibilities with business objectives.
            """),
            context=[],
            agent=agent,
            #callback=callback_function
            )
    def TechnologySystemsOverview_Task(self,agent):
        return Task(
            description=f"""
            Describe the necessary technology systems that underpin operational execution, including infrastructure, applications, and tools. Ensure these systems are designed to support the identified business processes and organizational structure efficiently.
            """,
            expected_output=dedent(f"""
            A clear and coherent organizational structure that aligns roles and responsibilities with business objectives.
            """),
            context=[],
            agent=agent,
            #callback=callback_function
            )
    def InformationFlowMechanism_Task(self,agent):
        return Task(
            description=f"""
            Propose mechanisms for managing data and knowledge across the organization to support decision-making and operations. These mechanisms should align with the business processes and utilize the technology systems efficiently.
            """,
            expected_output=dedent(f"""
            A proposal for information flow mechanisms that ensure timely, accurate, and efficient data and knowledge management.
            """),
            context=[],
            agent=agent,
            #callback=callback_function
            )
    def GovernanceCulture_Task(self,agent):
        return Task(
            description=f"""
            Develop guidelines for governance and decision-making processes that support strategic alignment and operational goals. Cultivate a supportive culture and leadership style that enhances organizational performance.
            """,
            expected_output=dedent(f"""
            Comprehensive guidelines for governance, culture, and leadership practices that support strategic alignment and operational efficiency.
            """),
            context=[],
            agent=agent,
            #callback=callback_function
            )
    def ImplementationRoadmap_Task(self,agent):
        return Task(
            description=f"""
            Create a detailed implementation roadmap for adopting the Operating Model, incorporating risk management and change management strategies. This roadmap should be actionable, with clear milestones and steps based on the comprehensive understanding of the Operating Model's components.
            """,
            expected_output=dedent(f"""
            A detailed implementation roadmap with steps, milestones, risk management, and change management strategies for adopting the Operating Model.
            """),
            context=[],
            agent=agent,
            #callback=callback_function
            )
    def CaseStudy_Task(self,agent):
        return Task(
            description=f"""
            Gather and analyze case studies of successful Operating Model implementations, highlighting best practices, challenges, and lessons learned. These case studies should serve as instructive examples for future implementations.
            """,
            expected_output=dedent(f"""
            A compilation of insightful case studies showcasing successful Operating Model implementations, with a focus on best practices and lessons learned.
            """),
            context=[],
            agent=agent,
            #callback=callback_function
            )

# Create a ContentCrew instance for the current part
parts = [f"Part {i+1}" for i in range(num_parts)]
scenes = [f"Scene {i+1}" for i in range(num_sections)] 
beats = [f"Beat {i+1}" for i in range(num_subsections)]

class ResearchCrew:
  def __init__(self,mycontent):
    self.mycontent = mycontent
    self.output_placeholder = st.empty()

  def run(self):
   #agents = ContentCreationAgents(mycontent)
    tasks = ContentCreation_tasks()
    
    research_agent = agents.lead_market_analyst(mycontent)
    
    market_research_task = tasks.market_analysis_task(research_agent)

    crew = Crew(
        agents=[research_agent],
        tasks=[market_research_task],
        verbose=True,
        Process=Process.hierarchical,
        manager_llm=LLM_selection
    )

    result = crew.kickoff()
    #self.output_placeholder.markdown(result)
    return result

Research_crew = ResearchCrew(mycontent)

class StrategeyCrew:
  def __init__(self,mycontent):
    self.mycontent = mycontent
    self.output_placeholder = st.empty()

  def run(self):
   #agents = ContentCreationAgents(mycontent)
    tasks = ContentCreation_tasks()
    
    strategy_agent = agents.chief_content_strategist(mycontent)    
    strategy_task = tasks.content_strategy_task(strategy_agent,part_idx)

    crew = Crew(
        agents=[strategy_agent],
        tasks=[strategy_task],
        verbose=True,
        Process=Process.sequential,
        manager_llm=LLM_selection
    )

    result = crew.kickoff()
    #self.output_placeholder.markdown(result)
    return result
  
  def refine(self):  
    #agents = ContentCreationAgents(mycontent)
    tasks = ContentCreation_tasks()
    
    strategy_agent = agents.chief_content_strategist(mycontent)    
    refine_task = tasks.refinestrategytask(strategy_agent,content_strategy)

    crew = Crew(
        agents=[strategy_agent],
        tasks=[refine_task],
        verbose=True,
        Process=Process.hierarchical,
        manager_llm=LLM_selection
    )

    result = crew.kickoff()
    #self.output_placeholder.markdown(result)
    return result

Strategy_crew = StrategeyCrew(mycontent)

class OutlineCrew:
  def __init__(self,mycontent):
    self.mycontent = mycontent
    self.output_placeholder = st.empty()

  def run(self):
   #agents = ContentCreationAgents(mycontent)
    tasks = ContentCreation_tasks()
    
    outline_agent = agents.chief_content_strategist(mycontent)    
    outline_task = tasks.content_strategy_task(outline_agent,part_idx)

    crew = Crew(
        agents=[outline_agent],
        tasks=[outline_task],
        verbose=True,
        Process=Process.sequential,
        manager_llm=LLM_selection
    )

    result = crew.kickoff()
    #self.output_placeholder.markdown(result)
    return result
  
  def refine(self):  
    #agents = ContentCreationAgents(mycontent)
    tasks = ContentCreation_tasks()
    
    outline_agent = agents.chief_content_strategist(mycontent)    
    refineoutline_task = tasks.refinestrategytask(outline_agent,content_strategy)

    crew = Crew(
        agents=[outline_agent],
        tasks=[refineoutline_task],
        verbose=True,
        Process=Process.hierarchical,
        manager_llm=LLM_selection
    )

    result = crew.kickoff()
    #self.output_placeholder.markdown(result)
    return result

Outline_crew = OutlineCrew(mycontent)
outline = None
class ContentCrew:
  def __init__(self,mycontent,part,scene,beat,outline):
    self.mycontent = mycontent
    self.part = part
    self.scene = scene
    self.beat = beat

    self.output_placeholder = st.empty()

  def run(self,mycontent,part,scene,beat,outline):
   #agents = ContentCreationAgents(mycontent)
    tasks = ContentCreation_tasks()
    
    content_agent = agents.creative_content_creator(mycontent)    
    content_task = tasks.create_content_task(content_agent,part,scene,beat,outline)

    crew = Crew(
        agents=[content_agent],
        tasks=[content_task],
        verbose=True,
        Process=Process.sequential,
        manager_llm=LLM_selection
    )

    result = crew.kickoff()
    #self.output_placeholder.markdown(result)
    return result
  
  def refine(self):  
    #agents = ContentCreationAgents(mycontent)
    tasks = ContentCreation_tasks()
    
    content_agent = agents.chief_content_strategist(mycontent)    
    contentrefine_task = tasks.refinestrategytask(content_agent,content_strategy)

    crew = Crew(
        agents=[content_agent],
        tasks=[contentrefine_task],
        verbose=True,
        Process=Process.hierarchical,
        manager_llm=LLM_selection
    )

    result = crew.kickoff()
    #self.output_placeholder.markdown(result)
    return result

Content_crew = ContentCrew(mycontent,part,scene,beat,outline)

# class ContentCrew():
#   def __init__(self, parts, scenes, beats):
#     self.parts = parts
#     self.scenes = scenes 
#     self.beats = beats
#     #self.get_outline = get_outline

#   def create(self, part, scene, beat):
#     #agents = ContentCreationAgents()
#     tasks = ContentCreation_tasks()
    
#     strategy_agent = agents.chief_content_strategist()
#     creator_agent = agents.creative_content_creator()
#     research_agent = agents.market_research_analyst()
    
#     #outline = self.get_outline(part, scene, beat)
#     task = tasks.create_content_task(part, scene, beat)

#     crew = Crew(
#         agents=[creator_agent],
#         tasks=[task],
#         verbose=True,
#         Process=Process.hierarchical,
#         manager_llm=LLM_selection
#     )
#     result = crew.kickoff()
#     return result

# content_crew = ContentCrew(parts, scenes, beats)

# Lead Market Analyst Agent
lead_market_analyst_solo = Agent(
    role='Lead Market Analyst',
    goal=f'Conduct in-depth market analysis and provide insights on content trends, list of audience pain points, list of opportunities to help address pain points, insightful audience characteristics, and market saturation of content to aid in creating targeted {content_type} content based on the following content idea: {mycontent}.',
    backstory="With a talent for uncovering hidden market trends and a deep understanding of content analysis, you have guided numerous clients to successfully cater to their ideal audience. Your analytical skills are unmatched, making you a cornerstone in the market research and analysis process.",
    tools=[
            SearchTools.search_internet,
            #BrowserTools.scrape_and_summarize_website,
            ContentTools().read_content
            ],
    llm=llm,
    memory=False,
    max_rpm=10,
    verbose=True,
    step_callback=streamlit_callback
    )

chief_content_strategist_solo = Agent(
    role='Chief Content Strategist',
    goal='Develop a content strategy based on market analysis done by the market analyst to guide the creative content creators content creation. Using all the most relevant information to create a cohesive list of compelling content ideas and topics relevant to audience and writing engaging content, your strategy must also include guidance for a provocative introduction, a conclusion and call to action.',
    backstory=f"""A visionary in content planning and strategy, chapter beats, creating content in the most engaging tone of the audience, brilliant and engaging style, you excel at transforming market insights of readers into actionable content plans. Your strategies are known for capturing audience interests and driving engagement.
    When you strategize content, you aim to not just impart information but also move people to think differently and take action. Your passion lies in using your gift of storytelling and incorporating the following guidelines:
    1. Begin with a Provocative Question or Statement: Start the {content_type} with a question or statement that immediately grabs attention. This could revolve around a common challenge or a controversial opinion in the field of agile metrics and analytics. The purpose is to provoke thought and stir the reader's curiosity from the very first sentence. 
    2. Incorporate a Personal Anecdote: Include a brief, engaging story that demonstrates the author's firsthand experience with the transformative power of agile metrics and analytics. This story should be relatable and highlight a specific instance where agile metrics significantly impacted an IT project's outcome. The narrative should showcase the author's expertise and the real-world application of the principles discussed in the {content_type}. 
    3. Outline the Promise of the content in the introduction only: Clearly articulate what the reader will gain by reading this {content_type} in the introduction. This section should highlight the unique insights, methodologies, or strategies the reader will learn and how this knowledge will empower them in their career or projects within the agile framework. It's important to make a compelling value proposition that promises practical benefits and newfound understanding. 
    4. Establish Credibility Briefly: Provide a concise statement of the author's credentials and experience in agile metrics and analytics. This should quickly assert the author's authority on the subject, reassuring the reader that the insights provided are rooted in professional expertise and success in the field. (Only Establish Credibilit in the introduction and conclusion)
    5. End each part with an Engaging Question or Interactive Element: Conclude the introduction with a rhetorical question or a prompt that encourages the reader to think actively about their own experiences or challenges with agile metrics. This could also be an invitation to engage with the {content_type}'s content in a way that feels personal and directly relevant to their own work or interests.
    """,
    llm=llm,
    memory=True,
    max_rpm=10,
    verbose=True,
    step_callback=streamlit_callback,
    #callbacks=[CustomStreamlitCallbackHandler(color="white")]
)

# Content Strategy Development Task
content_strategy_task_solo = Task(
    description=dedent(f"""Develop a content strategy incorporating market analysis insights, suggesting topics, tone, scene beats and sequence of content to make the {content_type} compelling for the target audience provided from the market analysis. Don't include word count estimates yet."""),
    expected_output="A detailed content strategy document guiding the creation of content should include the tone, scene beats, topic, subtopics, key points for each topic, and estimated word count for each section.",
    agent=chief_content_strategist_solo,
    #context=[market_analysis_task],
    #callback=callback_function
)
creative_content_creator_solo = Agent(
    role='Creative Content Creator',
    goal=dedent(f"""For each topic and sub-topic in the sequence and outline provided by the Chief Content Strategist, expand and write engaging and informative content for the {content_type}. The content should be written topic-wise and be coherent. 
                Each topic should be expanded into coherent sub-topics content and follow a logical sequence and include any additional considerations provided by the Chief Content Strategist. 
                Engage readers and encourage them to think critically about their own work environments and practices. This could involve questions posed directly to the reader, exercises at the end of chapters, or scenarios that invite readers to consider what they would do.
                When you develop content, you aim to not just impart information but also move people to think differently and take action. Your passion lies in using your gift of storytelling and incorporating the following guidelines:
    1. Begin with a Provocative Question or Statement: Start the {content_type} with a question or statement that immediately grabs attention. This could revolve around a common challenge or a controversial opinion in the field of agile metrics and analytics. The purpose is to provoke thought and stir the reader's curiosity from the very first sentence. 
    2. Incorporate a Personal Anecdote: Include a brief, engaging story that demonstrates the author's firsthand experience with the transformative power of agile metrics and analytics. This story should be relatable and highlight a specific instance where agile metrics significantly impacted an IT project's outcome. The narrative should showcase the author's expertise and the real-world application of the principles discussed in the {content_type}. 
    3. Outline the Promise of the content in the introduction only: Clearly articulate what the reader will gain by reading this {content_type} in the introduction. This section should highlight the unique insights, methodologies, or strategies the reader will learn and how this knowledge will empower them in their career or projects within the agile framework. It's important to make a compelling value proposition that promises practical benefits and newfound understanding. 
    4. Establish Credibility Briefly: Provide a concise statement of the author's credentials and experience in agile metrics and analytics. This should quickly assert the author's authority on the subject, reassuring the reader that the insights provided are rooted in professional expertise and success in the field. (Only Establish Credibilit in the introduction and conclusion)
    5. End each part with an Engaging Question or Interactive Element: Conclude the introduction with a rhetorical question or a prompt that encourages the reader to think actively about their own experiences or challenges with agile metrics. This could also be an invitation to engage with the {content_type}'s content in a way that feels personal and directly relevant to their own work or interests. \n
    Here's an example Introduction Paragraph:
    Have you ever wondered why, despite having access to a plethora of agile metrics, so many projects still veer off course? I found myself asking this question after witnessing project after project flounder, even with comprehensive dashboards at our disposal. It was a pivotal moment on a particularly troubled project that taught me the true value of interpreting and acting on these metrics correctly – not just as numbers, but as insights into the heart of project health. This {content_type} is your gateway to transforming agile metrics from mere data points into your most powerful project management tool. With my years of experience distilled into practical strategies and real-world examples, you're about to unlock the secrets to turning agile analytics into actionable wisdom.
    """),
    backstory="""A storyteller at heart, you blend creativity with insight to produce content that educates, entertains, and enlightens readers. Your written words have the power to captivate, inspire, and convey complex principles.
    """,
    llm=llm,
    memory=True,
    max_rpm=10,
    verbose=True,
    step_callback=streamlit_callback,
    #callbacks=[CustomStreamlitCallbackHandler(color="white")]
    )  

creation_crew_content_strategy = Crew(
    agents=[lead_market_analyst_solo, chief_content_strategist_solo],
    tasks=[content_strategy_task_solo],
    process=Process.hierarchical,
    #full_output=True,
    #step_callback=custom_step_callback,
    manager_llm=llm,
    function_call_llm=llm,
    verbose=True
)
# Content Creation Outline
content_creation_task_0_solo = Task(
    description=dedent(f"""Based on the input from the content strategist, Structure and write the {content_type}'s content outline in 5 distinct parts based on details provided by the Chief Content Strategist incorporating the content guidelines in the final content."""),
    expected_output=f"Outline only, based on the content strategy from the chief content strategist You will create 5 Parts, each part has {num_sections} sections, each section has {num_subsections} subsections, and at the end section are key takeaways for each subsection. The outline should follow the sequence and any additional considerations provided by the Chief Content Strategist in the content strategy.",
    agent=creative_content_creator_solo,
    context=[content_strategy_task_solo],
    #callback=callback_function
)
creation_crew_outline_solo = Crew(
    agents=[chief_content_strategist_solo, creative_content_creator_solo],
    tasks=[content_creation_task_0_solo],
    process=Process.hierarchical,
    #full_output=True,
    #step_callback=custom_step_callback,
    manager_llm=llm,
    function_call_llm=llm,
    verbose=True
)

#Streamlit UI Header
#st.image("./ContentCreationStudioIcon.png",width=250)
st.header("Content Creation Studio (Refactored)")
st.subheader("AI Writes, You Creat - The Perfect Duo",divider='rainbow')
# enter content
mycontent = st.text_area("Enter content creation idea here...", height=200)
if mycontent is not None:
    st.session_state['mycontent'] = mycontent
st.divider()
st.subheader("Content Structure")

# Initialize outline session state
for i in range(num_parts):
    st.session_state[f"user_outline_{i}"] = None
    st.session_state[f"result_outline_{i}"] = None

for part_idx in range(num_parts):
    st.session_state[f"result_strategy_{part_idx+1}"] = None

for part_idx in range(num_parts):
    st.session_state[f"result_outline_{part_idx+1}"] = None

# if content_type == "Operating Model":
#     outline_crew0 = OpModel_crew.outline(part_idx,mycontent)
# else: outline_crew0 = Outline_crew.run()

tab_names = ["Market Analysis", "Content Strategy", "Content Outline"] + [f"Part {i+1}" for i in range(num_parts)]
all_tabs = st.tabs(tab_names)

for i, tab_name in enumerate(tab_names):
    with all_tabs[i]:
        st.header(tab_name)
        if i == 0:
            #market analysis
            #with mna_tab:
            mna_col,mna_editor_col = st.columns(2)
            with mna_col:
                #st.subheader("Market Analysis")
                with st.expander("Click here to add your own Market Analysis"):
                    user_analysis = st.text_area("Add your market analysis here:", height=300)
                if st.button("🤖 Conduct Market Analysis",key="analysis") and mycontent is not None:
                    with st.status("CrewAI is working on Market Analysis for the {content_type}...",state="running",expanded=True) as status:
                        with st.container(height=500,border=False):
                            # Get your crew to work!
                            sys.stdout = StreamToExpander(st)
                            st.session_state['result_analysis'] = Research_crew.run()
                            status.update(label="✨ Mission Accomplished! Click here to see detailed AI messages",state="complete",expanded=False)
                st.markdown(st.session_state['result_analysis'])
            with mna_editor_col:
                st.subheader("Market Analysis Editor")
                mna_editor = st_quill(key="mna_editor")
        if i == 1:
            #Content Strategy
            with all_tabs[i]:
                part_tabs = st.tabs([f"Part {k+1}" for k in range(num_parts)])
                for part_idx, part_tab in enumerate(part_tabs):
                    with part_tab:
                        strategy_col,strategy_editor_col = st.columns(2)
                        with strategy_col:
                            st.subheader(f"Part {part_idx+1} Content Strategy")
                            with st.expander("Click here to add your own strategy"):
                                user_strategy = st.text_area("Add your own strategy here:", height=300,key=f"user_strategy_{part_idx+1}")
                                st.session_state["user_strategy"] = user_strategy
                            if st.button(f"🤖 Create Part {part_idx+1} Content Strategy",key= f"strategy_button_{part_idx+1}"):
                                with st.status(f"CrewAI is drafting Part {part_idx+1} Content Strategy...",state="running",expanded=True) as status:
                                    with st.container(height=500,border=False):
                                        # Get your crew to work!
                                        sys.stdout = StreamToExpander(st)
                                        generated_strategy = Strategy_crew.run()
                                        st.session_state['content_strategies'][part_idx] = generated_strategy
                                        status.update(label="✨ Mission Accomplished! Click here to see detailed AI messages",state="complete",expanded=False)
                            if part_idx < len(st.session_state['content_strategies']):
                                content_strategy = st.session_state['content_strategies'][part_idx]
                            if content_strategy is not None:
                                st.markdown(content_strategy)
                            if st.session_state["user_strategy"] is not None:
                                content_strategy = st.session_state["user_strategy"]
                            else:
                                content_strategy = st.session_state[f"result_strategy_{part_idx+1}"]
                            if st.session_state[f"result_strategy_{part_idx+1}"] is not None:
                                refine_strategy_input = st.text_area(label="Ask AI to refine the strategy:",placeholder="What refinements would you like AI to make to the strategy?", key= f"refine_strategy_{part_idx}") 
                                if refine_strategy_input is not None:
                                    if st.button("Refine Strategy",key=f"refine_strategy_button_{part_idx+1}"):
                                        with st.status("CrewAI is drafting the Content Strategy...",state="running",expanded=True) as status:
                                            with st.container(height=500,border=False):
                                                sys.stdout = StreamToExpander(st)
                                                #refined_strategy = 
                                                refined_strategy = Strategy_crew.refine()
                                                st.session_state['content_strategies'][part_idx] = refined_strategy
                                                status.update(label="✨ Mission Accomplished! Click here to see detailed AI messages",state="complete",expanded=False)
                                    st.markdown(content_strategy)
                #st.rerun()
                    with strategy_editor_col:
                        st.subheader("Content Strategy Editor")
                        strategy_editor = st_quill(key=f"strategy_editor_part_{part_idx}")
        if i == 2:
            #Content Outline
            with all_tabs[i]:
                part_tabs = st.tabs([f"Part {k+1}" for k in range(num_parts)])
                for part_idx, part_tab in enumerate(part_tabs):
                    with part_tab:
                        outline_col,outline_editor_col = st.columns(2)                    
                        with outline_col:
                            st.subheader(f"Part {part_idx+1} Content Outline")
                            with st.expander("Click here to add your own outline"):
                                user_outline = st.text_area("Add your own outline here:", height=300,key=f"user_outline_{part_idx+1}")
                                st.session_state["user_outline"] = user_outline
                            if st.button(f"🤖 Create Part {part_idx+1} Content Outline",key= f"outline_button_{part_idx+1}"):
                                with st.status(f"CrewAI is drafting Part {part_idx+1} Content Outline...",state="running",expanded=True) as status:
                                    with st.container(height=500,border=False):
                                        # Get your crew to work!                            
                                        sys.stdout = StreamToExpander(st)
                                        generated_outline = Outline_crew.run()
                                        st.session_state['content_outlines'][part_idx] = generated_outline
                                        status.update(label="✨ Mission Accomplished! Click here to see detailed AI messages",state="complete",expanded=False)
                            if part_idx < len(st.session_state['content_outlines']):
                                content_outline = st.session_state['content_outlines'][part_idx]
                            if content_outline is not None:
                                st.markdown(content_outline)
                            if st.session_state["user_outline"] is not None:
                                content_outline = st.session_state["user_outline"]
                            else:
                                content_outline = st.session_state[f"result_outline_{part_idx+1}"]
                            if st.session_state[f"result_outline_{part_idx+1}"] is not None:
                                refine_outline_input = st.text_area(label="Ask AI to refine the outline:",placeholder="What refinements would you like AI to make to the outline?", key= f"refine_outline_{part_idx}") 
                                if refine_outline_input is not None:
                                    if st.button("Refine Outline",key=f"refine_outline_button_{part_idx+1}"):
                                        with st.status("CrewAI is drafting the Content Outline...",state="running",expanded=True) as status:
                                            with st.container(height=500,border=False):
                                                sys.stdout = StreamToExpander(st)
                                                #refined_outline = 
                                                refined_outline = Outline_crew.refine()
                                                st.session_state['content_outlines'][part_idx] = refined_outline
                                                status.update(label="✨ Mission Accomplished! Click here to see detailed AI messages",state="complete",expanded=False)
                                    st.markdown(content_outline)
                #st.rerun()
                    with outline_editor_col:
                        st.subheader("Content Outline Editor")
                        outline_editor = st_quill(key=f"outline_editor_part_{part_idx}")

        if i >= 3:  # Part tabs
            with all_tabs[i]:
                part_content_tab,part_outline_tab, = st.tabs([f"{tab_name} Content",f"{tab_name} Outline"])
                with part_outline_tab:
                    st.subheader(f"{tab_name} Outline")
                    #with st.expander(f"Click to expand {tab_name} Outline"):
                    st.subheader("Content Outline")
                    user_outline = st.session_state.get(f"user_outline_{i}")
                    if not user_outline:
                        user_outline = st.text_area(f"Add your own {tab_name} Outline here:", key=f"part_user_outline_{i}")
                    outline = st.session_state.get(f"user_outline_{i}") or st.session_state.get(f"result_outline_{i}")
                    if st.button(f"🤖 Create {tab_name} Outline with AI", key=f"ai_outline_{i}"):
                        with st.status("CrewAI is drafting an Outline for the {content_type}...", state="running", expanded=True) as status:
                            with st.container(height=500, border=False):
                                # Get your crew to work!
                                sys.stdout = StreamToExpander(st)
                                st.session_state[f"result_outline_{i}"] = Outline_crew.run()
                                status.update(label="✨ Mission Accomplished! Click here to see detailed AI messages", state="complete", expanded=False)
                    if f"result_outline_{i}" in st.session_state:
                        st.markdown(st.session_state[f"result_outline_{i}"])
                        outline = user_outline or st.session_state[f"result_outline_{i}"]
                    else:
                        outline = user_outline
                        st.markdown(outline)
                    st.divider()

                with part_content_tab:
                    st.subheader(f"{tab_name} Content")
                    # Tabs for sections
                    section_tabs = st.tabs([f"Section {k+1}" for k in range(num_sections)])
                    for section_idx, section_tab in enumerate(section_tabs):
                        with section_tab:
                            # Expanders for subsections
                            for subsection_idx in range(num_subsections):
                                st.subheader(f"Sub-Section {subsection_idx+1}")
                                content_col,content_editor_col = st.columns(2)                    
                                with content_col:
                                    btn_count = 0
                                    btn_id = f"part{i-2}_section{section_idx+1}_subsection{subsection_idx+1}"
                                    unique_key = f"sub-{btn_id}"
                                    btn_count += 1
                                    if st.button(f"🤖 Create {btn_id}", key=unique_key):
                                        with st.status("🤖 CrewAI is working on the content...", state="running", expanded=True) as status:
                                            with st.container(height=500, border=False):
                                                # Get your crew to work!
                                                sys.stdout = StreamToExpander(st)
                                                generate_content = Content_crew.run(mycontent,i-2,section_idx+1,subsection_idx+1,outline)
                                                st.session_state['subsection_content'][subsection_idx] = generate_content
                                                status.update(label="✨ Mission Accomplished! Click here to see detailed AI messages", state="complete", expanded=False)
                                    if subsection_idx < len(st.session_state['subsection_content']):
                                        content_content = st.session_state['subsection_content'][subsection_idx]
                                    if content_content is not None:
                                        st.markdown(content_content)
                                with content_editor_col:
                                    st.subheader("Content Outline Editor")
                                    outline_editor = st_quill(key=f"content_editor_{btn_id}")
                                st.divider()
