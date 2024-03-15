from django.shortcuts import render
from app1.forms import *
import pickle
import pandas as pd
from Fruits_Vegs import settings
import os
import requests
from bs4 import BeautifulSoup
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import re

staticdir = settings.STATIC_DIR
print('--------')
print(staticdir)
print(os.listdir(staticdir))

#loading the Picle Files
print('--loading files-- ')
with open(os.path.join(staticdir, 'PickleFiles', 'classification_table.pkl'), 'rb') as file:
    cls = pickle.load(file)

with open(os.path.join(staticdir, 'PickleFiles', 'Classification_Model.pkl'), 'rb') as file:
    RF_CLF = pickle.load(file)

with open(os.path.join(staticdir, 'PickleFiles', 'similarity.pkl'), 'rb') as file:
    similarity = pickle.load(file)

with open(os.path.join(staticdir, 'PickleFiles', 'Prediction_Model.pkl'), 'rb') as file:
    LR = pickle.load(file)

with open(os.path.join(staticdir, 'PickleFiles', 'Labels.pkl'), 'rb') as file:
    LE = pickle.load(file)

with open(os.path.join(staticdir, 'PickleFiles', 'new_table.pkl'), 'rb') as file:
    new_table = pickle.load(file)

with open(os.path.join(staticdir, 'PickleFiles', 'pca.pkl'), 'rb') as file:
    pca = pickle.load(file)

with open(os.path.join(staticdir, 'PickleFiles', 'vitamins_info.pkl'), 'rb') as file:
    vitamins_info = pickle.load(file)

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
print('------completed loading files------')


#function for predicting calories and suggesting the similar items
def base(request):
    return render(request, 'app1/base.html')


def CaloriePrediction(request):
    if request.method == 'POST':
        form = nutrients_form(request.POST)
        if form.is_valid():
            dic = {
                'Calcium': form.cleaned_data['Calcium'],
                'Carbohydrate': form.cleaned_data['Carbohydrate'],
                'Cholesterol': form.cleaned_data['Cholesterol'],
                'Fatty_acids_total_saturated': form.cleaned_data['Fatty_acids_total_saturated'],
                'Fatty_acids_total_trans': form.cleaned_data['Fatty_acids_total_trans'],
                'Fiber': form.cleaned_data['Fiber'],
                'Iron': form.cleaned_data['Iron'],
                'Protein': form.cleaned_data['Protein'],
                'Sodium': form.cleaned_data['Sodium'],
                'Sugars': form.cleaned_data['Sugars'],
                'Fat': form.cleaned_data['Fat'],
            }

            energy = LR.predict(pd.DataFrame([dic]))
            energy = energy[0][0]
            dic['Energy'] = energy

            dic["Energy"] = energy
            ingredients = LE.inverse_transform(RF_CLF.predict(pca.transform(pd.DataFrame([dic]))))[0]
            index = cls[cls.ingredients == ingredients].index[0]
            similar_items = sorted(enumerate(similarity[index]), key=lambda x: x[1], reverse=True)[:10]
            li = [new_table.iloc[i].ingredients for i, _ in similar_items]

            content = {
                'form': nutrients_form(),
                'result': True,
                'Energy': energy,
                'predicted_item': ingredients,
                'similar_items': li,
            }
            return render(request, 'app1/CaloriePrediction.html', context=content)
    else:
        content = {'form': nutrients_form()}
        return render(request, 'app1/CaloriePrediction.html', context=content)



#based on the vitamin user enters need to suggest the food and giving extra information
def VitaminDeficiency(request):
    if request.method == "POST":
        form = Recommendation_form(request.POST)
        if form.is_valid():
            a = form.cleaned_data['Vitamin']
            url = vitamins_info[vitamins_info.Vitamin == a]['links'].iloc[0]
            response = requests.get(url)
            if response.status_code == 200:
                webpage_content = response.text
                soup = BeautifulSoup(webpage_content, 'html.parser')

                # Initialize food_items with an empty list
                food_items = []

                # Retrieving food to consume from the webpage content
                food_sources_heading = soup.find('h3', string='Food Sources')
                if not food_sources_heading:
                    food_sources_heading = soup.find('strong', string='Food Sources')

                if food_sources_heading:
                    ul_tag = food_sources_heading.find_next('ul')
                    if ul_tag:
                        list_items = ul_tag.find_all('li')
                        food_items = [item.get_text(strip=True) for item in list_items]
                    else:
                        print("No unordered list found after 'Food Sources' heading.")
                else:
                    print("Food Sources heading not found.")

                # Retreiving Signs of Deficiency and Toxicity Text from webpage contents
                def get_deficiency_toxicity_text(soup, tags):
                    for tag in tags:
                        deficiency_toxicity_tag = soup.find('h3', string=tag)
                        if deficiency_toxicity_tag:
                            deficiency_toxicity_text = ''
                            next_tag = deficiency_toxicity_tag.find_next_sibling()
                            while next_tag and next_tag.name != 'h3':
                                if next_tag.name == 'p':
                                    deficiency_toxicity_text += next_tag.get_text() + '\n'
                                elif next_tag.name == 'ul':
                                    for li in next_tag.find_all('li'):
                                        deficiency_toxicity_text += f"- {li.get_text()}\n"
                                next_tag = next_tag.find_next_sibling()
                            print(f'Signs of Deficiency and Toxicity Text (tag "{tag}"):')
                            return deficiency_toxicity_text.split('\n')
                    print('No matching <h3> tag found for Signs of Deficiency and Toxicity')

                tags_to_search = ['Signs of Deficiency and Toxicity', 'Signs of Deficiency', 'Signs of Deficiency and Toxicity ']
                deficiency_toxicity_text = get_deficiency_toxicity_text(soup, tags_to_search)

                # Summarizing the data from webpage content
                def split_text_into_chunks(text, chunk_size):
                    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                    return chunks

                def summarize_text(text, model, tokenizer):
                    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt", max_length=1024)
                    summary = model.generate(**tokens)
                    return tokenizer.decode(summary[0], skip_special_tokens=True)
                
                entry_content_div = soup.find('div', class_='entry-content')

                if entry_content_div:
                    main_content_text = entry_content_div.get_text(separator='\n').strip()
                    main_content_text = main_content_text.split("Related", 1)[0].strip()

                    # Split the text into chunks of 1000 words (adjust as needed)
                    chunk_size = 500
                    text_chunks = split_text_into_chunks(main_content_text, chunk_size)
                    summaries = [summarize_text(chunk, model, tokenizer) for chunk in text_chunks]
                    final_summary = ' '.join(summaries)
                else:
                    print('No <div> element with class "entry-content" found')
            else:
                print('Error fetching the webpage')

            dic = {'form': Recommendation_form(),
                    "food_items": food_items,
                    'deficiency_toxicity_text': deficiency_toxicity_text,
                    'final_summary': final_summary,
                    'diseases': vitamins_info[vitamins_info.Vitamin == a]['Disease'].values}

            return render(request, 'app1/VitaminDeficiency.html', context=dic)
    
    else:
        content = {'form': Recommendation_form()}
        return render(request, 'app1/VitaminDeficiency.html', context=content)
    


#recommending the food items based on the amount of calories
def Disease(request):
    if request.method == "POST":
        form =  Disease_form(request.POST)
        if form.is_valid():
            dis = form.cleaned_data['disease']

            def extract_important_text(title):
                url = f'https://en.wikipedia.org/wiki/{title}'
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Extract the main content of the page
                    main_content = soup.find('div', {'id': 'mw-content-text'})

                    if main_content:
                        important_text = ' '.join([p.get_text() for p in main_content.find_all('p')])
                        return important_text
                    else:
                        return 'Error: Main content not found on the Wikipedia page.'
                else:
                    return f'Error fetching the Wikipedia page. Status code: {response.status_code}'

            # Example usage
            disease_name = dis
            important_text = extract_important_text(disease_name)
            cleaned_text = re.sub(r'\[\d+\]', '', important_text)

            causes = []
            for i in [i.strip() for i in cleaned_text.split('.')]:
                if i.__contains__('cause'):
                    causes.append(i.replace('\n',''))
            Treatment = []
            for i in [i.strip() for i in cleaned_text.split('.')]:
                if i.__contains__('Treatment'):
                    Treatment.append(i.replace('\n',''))
            vitamin = vitamins_info[vitamins_info.Disease == dis]['Vitamin'].iloc[0]

            a = vitamin
            url = vitamins_info[vitamins_info.Vitamin == a]['links'].iloc[0]
            response = requests.get(url)
            if response.status_code == 200:
                webpage_content = response.text
                soup = BeautifulSoup(webpage_content, 'html.parser')

                #retrieving food to consume from the webpage content
                food_sources_heading = soup.find('h3', string='Food Sources')
                if food_sources_heading:
                    ul_tag = food_sources_heading.find_next('ul')
                    if ul_tag:
                        list_items = ul_tag.find_all('li')
                        food_items = [item.get_text(strip=True) for item in list_items]
                    else:print("No unordered list found after 'Food Sources' heading.")
                else:print("Food Sources heading not found.")


                #retreiving Signs of Deficiency and Toxicity Text from webpage contents
                def get_deficiency_toxicity_text(soup, tags):
                    for tag in tags:
                        deficiency_toxicity_tag = soup.find('h3', string=tag)
                        if deficiency_toxicity_tag:
                            deficiency_toxicity_text = ''
                            next_tag = deficiency_toxicity_tag.find_next_sibling()
                            while next_tag and next_tag.name != 'h3':
                                if next_tag.name == 'p':
                                    deficiency_toxicity_text += next_tag.get_text() + '\n'
                                elif next_tag.name == 'ul':
                                    for li in next_tag.find_all('li'):
                                        deficiency_toxicity_text += f"- {li.get_text()}\n"
                                next_tag = next_tag.find_next_sibling()
                            print(f'Signs of Deficiency and Toxicity Text (tag "{tag}"):')
                            return deficiency_toxicity_text.split('\n')
                    print('No matching <h3> tag found for Signs of Deficiency and Toxicity')

                tags_to_search = ['Signs of Deficiency and Toxicity', 'Signs of Deficiency', 'Signs of Deficiency and Toxicity ']
                deficiency_toxicity_text = get_deficiency_toxicity_text(soup, tags_to_search)
            
            
            dic = {'form': Disease_form(),
                   'causes':causes,
                   'Treatment':Treatment,
                   "food_items":food_items,
                   'deficiency_toxicity_text':deficiency_toxicity_text,
                   'diseases': vitamins_info[vitamins_info.Vitamin == a]['Disease'].values}
            
            return render(request,'app1/Disease.html',context = dic)
    
    else:
        content = {'form': Disease_form()}
        return render(request, 'app1/Disease.html', context=content)