from google import genai

def gemini_prompt_action(config, prompt):

    project = config['gcp_gemini_parameters']['project']
    model_name = config['gcp_gemini_parameters']['model_name']
    location = config['gcp_gemini_parameters']['location']

    client = genai.Client(vertexai=True, project=project, location=location)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    
    return response.text