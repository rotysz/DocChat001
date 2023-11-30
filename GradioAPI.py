from gradio_client import Client
#curl -X POST https://firmove-testy-chatcorpoapi.hf.space/run/predict -H 'Content-Type: application/json' -H 'Authorization: Bearer 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' -d '{"data": ["Jak sfinansowac zakup samochodu",0,"norm"]}'
HF_KEY = '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'
client = Client("Firmove-testy/ChatCorpoAPI",hf_token=HF_KEY)
client.view_api(all_endpoints=True)

sessions= []

session= 0

(result,session,sources) = client.predict(
		"Jak sfinansowac samochod?",
		session,
 	    "multiq",
		api_name="/predict"
)

print(result)
print(sources)
print(session)

(result,session,sources) = client.predict(
		"a maszyny budowlanej?",
		session,# str  in 'question' Textbox component
	 	    "multiq",
		api_name="/predict"
)
print(result)
print(sources)
print(session)