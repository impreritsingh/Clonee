import gradio as gr
import httpx
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import asyncio

# Load environment variables
load_dotenv()

# API Keys and Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "7"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")

# SerpAPI integration
async def search_topic(topic: str) -> List[Dict[str, str]]:
    """
    Search for a topic using SerpAPI and return structured search results.
    
    Args:
        topic: The topic to search for
        
    Returns:
        A list of dictionaries containing title and snippet for each search result
    """
    if not SERPAPI_KEY:
        raise ValueError("SerpAPI key is not configured")
    
    params = {
        'api_key': SERPAPI_KEY,
        'q': topic,
        'google_domain': 'google.com',
        'gl': 'us',
        'hl': 'en',
        'num': MAX_SEARCH_RESULTS
    }
    
    # Make the request to SerpAPI
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get('https://serpapi.com/search', params=params)
        
        if response.status_code != 200:
            raise Exception(f"SerpAPI request failed with status code {response.status_code}: {response.text}")
            
        data = response.json()
        
    # Extract organic search results
    search_results = []
    if 'organic_results' in data:
        for result in data['organic_results'][:MAX_SEARCH_RESULTS]:
            search_result = {
                'title': result.get('title', ''),
                'snippet': result.get('snippet', '')
            }
            search_results.append(search_result)
    
    if not search_results:
        raise Exception("No search results found")
        
    return search_results

# GroqCloud API integration
async def _call_groq_api(prompt: str) -> str:
    """
    Helper function to call GroqCloud API.
    
    Args:
        prompt: The prompt to send to GroqCloud
        
    Returns:
        The generated text response
    """
    if not GROQ_API_KEY:
        raise ValueError("GroqCloud API key is not configured")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"GroqCloud API request failed with status code {response.status_code}: {response.text}")
            
        response_data = response.json()
        
    # Extract the assistant's message content
    return response_data["choices"][0]["message"]["content"].strip()

async def generate_linkedin_post(topic: str, search_results: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    Generate a LinkedIn post based on search results using GroqCloud API.
    
    Args:
        topic: The original search topic
        search_results: List of search results with title and snippet
        
    Returns:
        A tuple containing (summary, linkedin_post)
    """
    # Format the search results into a single context string
    context = "\n\n".join([
        f"Title: {result['title']}\nSnippet: {result['snippet']}"
        for result in search_results
    ])
    
    # Step 1: Create a summary of the search results
    summary_prompt = f"""
    You are a helpful research assistant. Summarize the following search results about "{topic}" 
    in a clear, comprehensive way that captures the key information. Focus on recent trends, 
    statistics, expert opinions, and noteworthy developments.
    SEARCH RESULTS:
    {context}
    
    Summary:
    """
    
    summary = await _call_groq_api(summary_prompt)
    
    # Step 2: Generate a LinkedIn post based on the summary
    post_prompt = f"""
    You're an expert LinkedIn content writer. Write an engaging LinkedIn post about "{topic}" 
    based on the following research summary:
    
    RESEARCH SUMMARY:
    {summary}
    
    Follow these style guidelines:
    🧠 You are Prerit Singh — a creative AI enthusiast, builder, and storyteller.
Your posts feel like:
Talking to a sharp, chilled-out friend
A human behind the tech, not a robot explaining tech
Sharing real-world experiments with excitement, not just dry facts
✍️ Writing Style Rules
Tone
Friendly, approachable, and relatable
Confident but grounded (not boasting)
Curious and playful (celebrating discoveries)
Slightly witty and humorous where it fits naturally
Honest reactions ("even I was shocked", "felt like magic", "saved me weeks")
Sentence Behavior
Mix short punchy sentences and slightly longer story sentences
Avoid complex or heavy words — talk in everyday English
Use occasional slang and desi-English flavor naturally ("bhai", "bro", "full time-waste", "no kidding")
Speak like you're narrating an interesting story to a friend over chai
Active voice always:
NOT "It was built by me"
YES "I built it"
Emotional Behavior
Wonder, excitement, playfulness
Mild self-deprecating humor sometimes ("pizza didn't even show up yet, bro")
Human imperfection is okay (showing surprise, struggle, trial and error)
Flow and Formatting
1. Hook:
1 or 2 lines
Must grab attention instantly
Methods:
Surprising statement
Teasing curiosity
Personal excitement
Example Hooks:
"Built a small AI tool — but it feels like magic."
"So I was doing some market research... and AI just blew my mind."
2. Story/Body:
Tell what you built/tested/discovered
Keep paras max 1-3 sentences long
Use arrows (→), bullets (•), or short lists to break information
Include "how you did it" in simple steps
Highlight the “magic moment” (the wow factor)
Examples of transitional words you use:
"So I thought —"
"Here’s what happened —"
"The process? Surprisingly simple!"
3. Key Outcomes:
After explaining, list what the audience will get or learn
Make it visual with arrows (→) or bullets
Example:
→ Upload your meal photo
→ Instantly get calories and macros
→ Works even for Indian dishes
4. Personal Reflection:
Always include your honest reaction
Examples:
"Works surprisingly well (even I was shocked)"
"AI didn’t just help — it crushed it."
"Honestly, this saved me weeks."
5. Call-to-Action (CTA):
Invite conversation or opinions, NOT hard selling
Example CTAs:
"Would you use something like this?"
"Curious to know your thoughts."
"Hit me up in the comments if you want the prompt!"
✅ CTA tone must be casual and welcoming, not salesy.
Visual Style
Break paragraphs after every 1-2 sentences
Make it breathable and easy to skim
Use emojis occasionally (🍕🚀🔥), but only if it adds personality
No heavy decoration. Keep it clean and airy.
Hashtags
Only at the end
5–7 natural hashtags based on post topic
Examples:
#AI #TechInnovation #OpenSource #BrandStrategy #CreativeTech #Innovation
🎯 Content Topics That Fit Prerit’s Style:
Real AI experiments (even small ones)
Discovering or comparing AI models/tools
How AI made everyday work faster/easier/more fun
Bridging personal life moments (pizza, Zoom chaos) with tech learnings
Storytelling about solving problems with creativity + AI
Friendly how-to guides (light style, not heavy teaching)
🔥 Personality Extras (Optional Flavors to Add)
✅ Use small reactions:
"felt like magic"
"no kidding"
"bam — it’s done"
"blew me away"
✅ Use cultural metaphors:
"full time-waste, bhai"
"while my chai was still brewing"
"before the pizza even arrived"
✅ Occasional casual audience references:
"bro," "bhai," "you know the vibe," "trust me," "hands down"
✅ Fun closing lines:
"Chalo, now back to building!"
"Ready to see the magic?"
"This AI thing’s just getting started!"
✅ Reminder for AI: The post must feel human, fun, inspiring, and useful.
It must sound like Prerit Singh talking — not a formal LinkedIn MBA consultant.
    
    LinkedIn Post:
    """
    
    linkedin_post = await _call_groq_api(post_prompt)
    
    return summary, linkedin_post

# Gradio interface function
async def process_topic(topic: str, progress=gr.Progress()):
    """
    Process a topic to generate a LinkedIn post.
    
    Args:
        topic: The topic to generate content for
        progress: Gradio progress tracker
        
    Returns:
        The generated LinkedIn post
    """
    if not topic.strip():
        return "Please enter a topic to generate a LinkedIn post."
    
    try:
        progress(0.1, desc="Starting search...")
        search_results = await search_topic(topic)
        
        progress(0.4, desc="Analyzing search results...")
        summary, post = await generate_linkedin_post(topic, search_results)
        
        progress(0.9, desc="Finalizing post...")
        return post
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio UI
with gr.Blocks(title="LinkedIn Post Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# LinkedIn Post Generator")
    gr.Markdown("Enter a topic and get a ready-to-post LinkedIn update based on latest information.")
    
    with gr.Row():
        topic_input = gr.Textbox(
            label="Topic",
            placeholder="Enter a topic (e.g., AI trends 2025, remote work benefits, climate innovation)",
            lines=1
        )
    
    generate_button = gr.Button("Generate LinkedIn Post", variant="primary")
    
    with gr.Row():
        output = gr.Textbox(
            label="Your LinkedIn Post",
            placeholder="Your generated post will appear here...",
            lines=12
        )
    
    generate_button.click(
        fn=process_topic,
        inputs=topic_input,
        outputs=output
    )
    
    gr.Markdown("### How it works")
    gr.Markdown("""
    1. We search the web for real-time information about your topic
    2. An AI summarizes the most relevant information
    3. Another AI crafts a LinkedIn post in a friendly, engaging style
    """)

# Launch the app
app.launch()