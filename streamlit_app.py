from google.cloud import aiplatform
import vertexai
from vertexai.preview.language_models import TextGenerationModel
import pandas as pd
from enum import Enum
vertexai.init(project="laxmanpersonal", location="us-central1")
model = TextGenerationModel.from_pretrained("text-bison@001")
def predict_large_language_model_sample(
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    ) :
    """Predict using a Large Language Model."""
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    # print(f"Response from Model: {response.text}")
    return response

prompt_template_0 = """Act as an expert care manager specialist supporting wellbeing and chronic conditions programs, you are talking to a member about their health conditions need to deliver the below message in quotes in an efficient personalized way to match their personal characteristics ,beliefs and socio economic conditions.

"%s"

Rewrite the above message in quotes for a member in an efficient personalized way to match their personal characteristics , beliefs and socio economic conditions . Include relevant quotes and health content so it is for the member to understand and trust
%s
"""
print(prompt_template_0)
cc_text=[
    "Being at a healthy weight can reduce your risk for diabetes, high blood pressure, stroke, and heart disease. Regular exercise and healthy eating can help you work toward and maintain a healthy weight. To get to a healthier weight, work with your health care provider to create a weight loss plan",
    "Everyone should get the flu shot.The flu shot protects against both seasonal and H1N1 flu.The best time to get the flu shot is October through December.But you can still prevent the flu by getting a shot in January or February.Talk to your doctor about the flu shot.",
    "Healthy eating can help you lower your risk for heart disease, high blood pressure, high cholesterol, and other health problems. You can start eating healthier by taking small steps. For instance, eat five or more servings of fruits and vegetables a day, limit the fatty foods and sweets you eat, and lower your salt intake. Talk to your health care provider to learn more about a healthy diet."
]

cc_title=[
    'CC_Lose_weight', 
    'CC_Take_a_flushot', 
    'CC_Eat_healthy'
]
class Age(Enum):
    YOUNG='a young person'
    OLD='a old person'
    
class Gender(Enum):
    MALE='a male'
    FEMALE='a female'
    OTHER='other'

class Engagement(Enum):
    LOW='Low level of engagement with health & wellness programs'
    MEDIUM='Medium level of engagement with health & wellness programs'
    HIGH='High level of engagement with health & wellness programs'

class Trust(Enum):
    LOW_TRUST='Seek second opinions and multiple trusted, highly-recommended sources prior to making decisions'
    HIGH_TRUST='Not looking for multiple opinions and multiple data points.'
    EXPERT='Care about finding experts who understand and reliably support their specific health needs'
    
class HealthFoodAccess(Enum):
    NO='No access to healthy food'
    YES='Has access to healthy food'
    
class EconomicCondition(Enum):
    POOR='Poor economic condition'
    GOOD='Good economic condition'
    
class Education(Enum):
    NO_DEGREE='Not have any degree'
    HIGH_SCHOOL_DEGREE='Has a high school degree'
    BACHELOR_DEGREE='Has a bachelor degree'
    MASTER_DEGREE='Has a master degree'
    DOCTORATE_DEGREE='Has a PhD degree'




    
def getUserText(userattrs):
    s=""
    for attr in userattrs:
        s+='- '+attr.value+'\n'
    return s
### Personalize by Age
df = pd.DataFrame(columns = ['Member']+cc_title)
users = [
    [Age.YOUNG,Gender.MALE,Engagement.HIGH, Trust.EXPERT],
    [Age.OLD,Gender.MALE,Engagement.HIGH, Trust.EXPERT]
]
for user in users:
    
    row = {
        'Member' : user[0].value,
        cc_title[0] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[0],getUserText(user))).text,
        cc_title[1] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[1],getUserText(user))).text,
        cc_title[2] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[2],getUserText(user))).text
    }
    df=df.append(row, ignore_index=True)
# df
print(prompt_template_0%(cc_text[0],getUserText(users[0])))
pd.set_option('display.max_colwidth', None)
df[['Member',cc_title[0]]]
df[['Member',cc_title[1]]]
df[['Member',cc_title[2]]]
Personalize by Engagement
df = pd.DataFrame(columns = ['Member']+cc_title)
users = [
    [Age.YOUNG,Gender.MALE,Engagement.HIGH, Trust.EXPERT],
    [Age.YOUNG,Gender.MALE,Engagement.LOW, Trust.EXPERT]
]
for user in users:
    
    row = {
        'Member' : user[2].value,
        cc_title[0] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[0],getUserText(user))).text,
        cc_title[1] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[1],getUserText(user))).text,
        cc_title[2] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[2],getUserText(user))).text
    }
    df=df.append(row, ignore_index=True)
df[['Member',cc_title[0]]]
df[['Member',cc_title[1]]]
df[['Member',cc_title[2]]]
### Personalize by Health Food Access
df = pd.DataFrame(columns = ['Member']+cc_title)
users = [
    [Age.YOUNG,Gender.MALE,Engagement.HIGH,Trust.EXPERT, HealthFoodAccess.NO],
    [Age.YOUNG,Gender.MALE,Engagement.HIGH, Trust.EXPERT, HealthFoodAccess.YES]
]
for user in users:
    
    row = {
        'Member' : user[2].value,
        cc_title[0] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[0],getUserText(user))).text,
        cc_title[1] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[1],getUserText(user))).text,
        cc_title[2] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[2],getUserText(user))).text
    }
    df=df.append(row, ignore_index=True)
df[['Member',cc_title[0]]]
df[['Member',cc_title[1]]]
df[['Member',cc_title[2]]]
### Personalize by Economic Condition
df = pd.DataFrame(columns = ['Member']+cc_title)
users = [
    [Age.YOUNG,Gender.MALE,Engagement.HIGH,Trust.EXPERT, HealthFoodAccess.YES, EconomicCondition.POOR],
    [Age.YOUNG,Gender.MALE,Engagement.HIGH, Trust.EXPERT, HealthFoodAccess.YES, EconomicCondition.GOOD]
]
for user in users:
    
    row = {
        'Member' : user[2].value,
        cc_title[0] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[0],getUserText(user))).text,
        cc_title[1] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[1],getUserText(user))).text,
        cc_title[2] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[2],getUserText(user))).text
    }
    df=df.append(row, ignore_index=True)
df[['Member',cc_title[0]]]
df[['Member',cc_title[1]]]
df[['Member',cc_title[2]]]
### Personalize by Education
df = pd.DataFrame(columns = ['Member']+cc_title)
users = [
    [Age.YOUNG,Gender.MALE,Engagement.HIGH,Trust.EXPERT, HealthFoodAccess.YES, EconomicCondition.GOOD, Education.NO_DEGREE],
    [Age.YOUNG,Gender.MALE,Engagement.HIGH, Trust.EXPERT, HealthFoodAccess.YES, EconomicCondition.GOOD, Education.BACHELOR_DEGREE]
]
for user in users:
    
    row = {
        'Member' : user[2].value,
        cc_title[0] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[0],getUserText(user))).text,
        cc_title[1] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[1],getUserText(user))).text,
        cc_title[2] : predict_large_language_model_sample(0.2, 512, 0.8, 1, prompt_template_0%(cc_text[2],getUserText(user))).text
    }
    df=df.append(row, ignore_index=True)
df[['Member',cc_title[0]]]
df[['Member',cc_title[1]]]
df[['Member',cc_title[2]]]
