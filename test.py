import google.generativeai as genai

genai.configure(api_key="AIzaSyChD-XPFVlDho8IerlZxxmRlCfjJmg4-Z4")

# Create the model
model = genai.GenerativeModel('gemini-1.5-flash')

# Send a prompt
response = model.generate_content("What is the capital of Sweden?")

print(response.text)