import ell
import os
from openai import OpenAI
from typing import List

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"])
ell.init(store='./logdir', autocommit=True, verbose=True)

@ell.simple(model="gpt-4o-mini", temperature=1.0, client=client)
def generate_story_ideas(about : str):
    """You are an expert story ideator. Only answer in a single sentence."""
    return f"Generate a story idea about {about}."

@ell.simple(model="gpt-4o-mini", temperature=1.0, client=client)
def write_a_draft_of_a_story(idea : str):
    """You are an adept story writer. The story should only be 3 paragraphs."""
    return f"Write a story about {idea}."

@ell.simple(model="gpt-4o-mini", temperature=1.0, client=client)
def choose_the_best_draft(drafts : List[str]):
    """You are an expert fiction editor."""
    return f"Choose the best draft from the following list: {' '.join(drafts)}."

@ell.simple(model="gpt-4o-mini", temperature=0.2, client=client)
def write_a_really_good_story(about : str):
    """You are an expert novelist that writes in the style of Hemmingway. You write in lowercase."""
    # Note: You can pass in api_params to control the language model call
    # in the case n = 4 tells OpenAI to generate a batch of 4 outputs.
    ideas = generate_story_ideas(about, api_params=(dict(n=4)))

    drafts = [write_a_draft_of_a_story(idea) for idea in ideas]

    best_draft = choose_the_best_draft(drafts)


    return f"Make a final revision of this story in your voice: {best_draft}."

story = write_a_really_good_story("a dog")