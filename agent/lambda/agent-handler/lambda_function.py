import json
import datetime
import time
import os
import dateutil.parser
import logging
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage
from chat import Chat
from fsi_agent import FSIAgent
import pdfrw
import difflib


# Instantiate boto3 clients and resources
boto3_session = boto3.Session(region_name=os.environ['AWS_REGION'])
bedrock_client = boto3_session.client(service_name="bedrock-runtime")

# --- Lex v2 request/response helpers (https://docs.aws.amazon.com/lexv2/latest/dg/lambda-response-format.html) ---

# This function handles formatting responses back to Lex.
def lex_format_response(event, response_text):
    event['sessionState']['intent']['state'] = "Fulfilled"

    return {
        'sessionState': {
            'sessionAttributes': {'history': 'none'},
            'dialogAction': {
                'type': 'Close'
            },
            'intent': event['sessionState']['intent']
        },
        'messages': [{'contentType': 'PlainText','content': response_text}],
        'sessionId': event['sessionId'],
        'requestAttributes': event['requestAttributes'] if 'requestAttributes' in event else None
    }

def elicit_intent(intent_request, session_attributes, message):
    response = {
        'sessionState': {
            'dialogAction': {
                'type': 'ElicitIntent'
            },
            'sessionAttributes': session_attributes
        },
        'messages': [
            {
                'contentType': 'PlainText', 
                'content': message
            },
            {
                'contentType': 'ImageResponseCard',
                'imageResponseCard': {
                    "buttons": [
                        {
                            "text": "Loan Application",
                            "value": "Loan Application"
                        },
                        {
                            "text": "Loan Calculator",
                            "value": "Loan Calculator"
                        },
                        {
                            "text": "Ask GenAI",
                            "value": "What kind of questions can the Assistant answer?"
                        }
                    ],
                    "title": "How can I help you?"
                }
            }     
        ]
    }

    return response

def invoke_fm(prompt):
    """
    Invokes Foundational Model endpoint hosted on Amazon Bedrock and parses the response.
    """
    chat = Chat(prompt)
    llm = Bedrock(client=bedrock_client, model_id="anthropic.claude-v2", region_name=os.environ['AWS_REGION']) # "anthropic.claude-instant-v1"
    llm.model_kwargs = {'max_tokens_to_sample': 350}
    lex_agent = FSIAgent(llm, chat.memory)
    formatted_prompt = "\n\nHuman: " + prompt + " \n\nAssistant:"

    try:
        message = lex_agent.run(input=formatted_prompt)
    except ValueError as e:
        message = str(e)
        if not message.startswith("Could not parse LLM output:"):
            raise e
        message = message.removeprefix("Could not parse LLM output: `").removesuffix("`")

    return message

def genai_intent(intent_request):
    """
    Performs dialog management and fulfillment for user utterances that do not match defined intents (i.e., FallbackIntent).
    Sends user utterance to Foundational Model endpoint via 'invoke_fm' function.
    """
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    
    if intent_request['invocationSource'] == 'DialogCodeHook':
        prompt = intent_request['inputTranscript']
        output = invoke_fm(prompt)
        return elicit_intent(intent_request, session_attributes, output)

def kendra_search(intent_request):
    """
    Primary function that performs a kendra search, using the retrieve API and passes the kendra response into the
    invoke LLM function.
    :param question: The question the user inputs within app.py or the frontend
    :return: Returns the final response of the LLM that was created by the invokeLLM function
    """
    kendra = boto3.client('kendra',region_name='us-east-1')
    index_id = "823fed26-38f9-490a-bdfc-d89e19f95a63"
    kendra_response = kendra.retrieve(
        IndexId=index_id,  
        QueryText="get me wiki",
        PageNumber=1,
        PageSize=15
    )
    
    return invoke_llm(intent_request, kendra_response)

def invoke_llm(intent_request, kendra_response):
    """
    This function takes in the question from the user, along with the Kendra responses as context to generate an answer
    for the user on the frontend.
    :param question: The question the user is asking that was asked via the frontend input text box.
    :param kendra_response: The response from the Kendra document retrieve query, used as context to generate a better
    answer.
    :return: Returns the final answer that will be provided to the end-user of the application who asked the original
    question.
    """
    
    
    bedrock = boto3.client('bedrock-runtime', 'us-east-1', endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com')
    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'
    # prompt that is passed into the LLM with the Kendra Retrieval context and question
    # TODO: FEEL FREE TO EDIT THIS PROMPT TO CATER TO YOUR USE CASE
    prompt_data = f"""\n\nHuman:    
Answer the following question in a manner andonly if it can be answered from the Kendra index created.
Do not include information that is not relevant to the question.
Only provide information based on the data available with you and do not make assumptions
Avoid all the non related questions to Echo business
Use the provided examples as reference
###
Here is an example
<example>
Question: What is Kendra
Assistant: I dont know
</example>
###
###
Here is an example
<example>
Question: Give me the list of all available wiki documents?
Assistant: Sure, Let me search and collect the information
</example>
###
###
Here is an example
<example>
Question: How many different documents do you have?
Assistant: 500+
</example>
###
Question: How many different authors do you find?
Assistant: I dont know

Question: {intent_request}

Context: {kendra_response}

###

\n\nAssistant:

"""
    # body of data with parameters that is passed into the bedrock invoke model request
    # TODO: TUNE THESE PARAMETERS AS YOU SEE FIT
    body = json.dumps({"prompt": prompt_data,
                       "max_tokens_to_sample": 8191,
                       "temperature": 0,
                       "top_k": 250,
                       "top_p": 0.5,
                       "stop_sequences": []
                       })
    # Invoking the bedrock model with your specifications
    response = bedrock.invoke_model(body=body,
                                    modelId=modelId,
                                    accept=accept,
                                    contentType=contentType)
    
    response_body = json.loads(response.get('body').read())
    answer = response_body.get('completion')
    return answer
        
# --- Main handler ---

def handler(event, context):
    user_input = event['inputTranscript']
    if user_input == '':
        response_text = kendra_search(user_input)    
    else:
        response_text = genai_intent(event)
    """
    Invoked when the user provides an utterance that maps to a Lex bot intent.
    The JSON body of the user request is provided in the event slot.
    """
    os.environ['TZ'] = 'America/New_York'
    time.tzset()

    return lex_format_response(event,response_text)