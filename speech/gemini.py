from google import genai

# upload file 

client = genai.Client()

myfile = client.files.upload(file='media/sample.mp3')

response = client.models.generate_content(
  model='gemini-2.0-flash',
  contents=['Describe this audio clip', myfile]
)

print(response.text)


# get metadata

myfile = client.files.upload(file='media/sample.mp3')
file_name = myfile.name
myfile = client.files.get(name=file_name)
print(myfile)


# list uploaded files

print('My files:')
for f in client.files.list():
    print(' ', f.name)


# delete uploaded files 
myfile = client.files.upload(file='media/sample.mp3')
client.files.delete(name=myfile.name)