import os
import pandas as pd
import numpy as np
from unidecode import unidecode

def grab_reval_borders():
    file_path = "data/input_form.csv"
    df = pd.read_csv(file_path)

    na_border = []
    code_to_border = {}
    for _, row in df.iterrows():
        story_code = row['Story Code']
        reveal_border = row['What is the "reveal border" sentence?']
        
        if type(reveal_border) == float:
            na_border.append(story_code)
            continue

        code_to_border[story_code] = unidecode(reveal_border)

    return code_to_border, na_border

def prep_partitions_and_plain_text(code_to_border, story_codes, dir_path):
    # Start paritioning the story by reveal border
    code_to_partitions = {}
    code_to_plain_text = {}

    no_border_list = []
    for story_code, file_name in story_codes.items():
        file = open(os.path.join(dir_path, file_name), 'r')
        plain_text_list = []
        for line in file.readlines():
            plain_text_list.append(line.strip())
        plain_text = unidecode(' '.join(plain_text_list))

        splitted_plain_text = plain_text.split(code_to_border[story_code])
        if len(splitted_plain_text) == 2:
            # Find the sentence successfully, and partition them into twice.
            code_to_partitions[story_code] = {"first": splitted_plain_text[0], "second": splitted_plain_text[1]}
            code_to_plain_text[story_code] = plain_text
        else:
            # 1. Deal with annotation error case 
            # The annotators add an additional quatation mark at the end of border sentence, while these sentences do not have quatation mark in the original text.
            # These can exlucde around 20 cases out of 32.
            if code_to_border[story_code][-1] == "\"":
                # Correct the border sentence
                code_to_border[story_code] = code_to_border[story_code][:-1]

                # Find the border sentence again
                splitted_plain_text = plain_text.split(code_to_border[story_code])
                if len(splitted_plain_text) == 2:
                    code_to_partitions[story_code] = {"first": splitted_plain_text[0], "second": splitted_plain_text[1]}
                    code_to_plain_text[story_code] = plain_text
                    continue
                else:
                    code_to_border[story_code] = code_to_border[story_code] + "\""

                # Reduce the beginning and last qutations
                if story_code in ["OMIC11", 'ATT15', 'ATT13', 'CKS02', 'PZ01', 'CKS05', 'UAMM13']:
                    code_to_border[story_code] = code_to_border[story_code][1:-1]

                    splitted_plain_text = plain_text.split(code_to_border[story_code])
                    code_to_partitions[story_code] = {"first": splitted_plain_text[0], "second": splitted_plain_text[1]}
                    code_to_plain_text[story_code] = plain_text
                    continue
                
                # Other cases are specific to different story
                if story_code == "UAMM12":
                    code_to_border[story_code] = "\"" + unidecode(code_to_border[story_code])[:-1]

                    splitted_plain_text = plain_text.split(code_to_border[story_code])
                    code_to_partitions[story_code] = {"first": splitted_plain_text[0], "second": splitted_plain_text[1]}
                    code_to_plain_text[story_code] = plain_text
                    continue
                
                if story_code == "CKS08":
                    code_to_border[story_code] = unidecode("\"Now,\" resumed Kennedy, his tone changing, \"suppose we try a little experiment—one that was tried very convincingly by the immortal Liebig.")

                    splitted_plain_text = plain_text.split(code_to_border[story_code])
                    code_to_partitions[story_code] = {"first": splitted_plain_text[0], "second": splitted_plain_text[1]}
                    code_to_plain_text[story_code] = plain_text
                    continue
                
                if story_code == "CBSH03":
                    code_to_border[story_code] = "\"Where is this Count Sylvius?\""

                    splitted_plain_text = plain_text.split(code_to_border[story_code])
                    code_to_partitions[story_code] = {"first": splitted_plain_text[0], "second": splitted_plain_text[1]}
                    code_to_plain_text[story_code] = plain_text
                    continue

                if story_code == 'JTC01':
                    code_to_border[story_code] = "The first witness for the defence was Thorndyke;"

                    splitted_plain_text = plain_text.split(code_to_border[story_code])
                    code_to_partitions[story_code] = {"first": splitted_plain_text[0], "second": splitted_plain_text[1]}
                    code_to_plain_text[story_code] = plain_text
                    continue

                if story_code == 'TGS06':
                    code_to_border[story_code] = "\"You have a story to tell. Will you tell it to me? It may save Miss Postlethwaite's life.\""

                    splitted_plain_text = plain_text.split(code_to_border[story_code])
                    code_to_partitions[story_code] = {"first": splitted_plain_text[0], "second": splitted_plain_text[1]}
                    code_to_plain_text[story_code] = plain_text
                    continue

                if len(splitted_plain_text) != 2:
                    no_border_list.append(story_code)
            else:
                no_border_list.append(story_code)

            
    # nlp = spacy.load("en_core_web_sm")

    # tokenized = nlp(plain_text, disable=["parser", "ner"])
    # tokenized_text = [token.text for token in tokenized]
    # print(tokenized_text)
    # exit()
    
    really_no_border = ['PVDS41', 'CBSH05', 'CKS53', 'OMIC03', 'ASH09', 'OMIC04'] 
    # PVDS41: From the input_form.csv dataset, it seems like PVDS41 has the same reveal border as PVDS40. I guess that's a typo.
    # CBSH05: From the input_form.csv dataset, there are two CBSH05 with completely different annotation content. I guess that's a typo.
    # CKS53: Can't find the sentence. Please double check what happened.
    # OMIC03: Can't find the sentence. Please double check what happened.
    # ASH09: From the input_form.csv dataset, it seems like ASH09 has the same reveal border as ASH12. I guess that's a typo.
    # OMIC03: Can't find the sentence. Please double check what happened.

    # for story_code in no_border_list:
    #     print(story_code)
    #     print(code_to_border[story_code])
    #     print("\n")
    return code_to_partitions, code_to_plain_text, no_border_list


def preprocessing():
    print("-----Start doing preprocessing-----")
    code_to_border, na_border = grab_reval_borders()
    print("These stories don't have a reveal border", na_border)

    dir_path = 'data/plain_texts'
    file_names = [f for f in os.listdir(dir_path)]

    story_codes = {file_name.split(' ')[0]: file_name for file_name in file_names}

    file_without_annotations = {}
    for story_code, file_name in story_codes.items():
        if story_code not in code_to_border and story_code not in na_border:
            file_without_annotations[story_code] = file_name

    print("\nThese stories are not annotated in input_form.csv", file_without_annotations)
    print("Note, TEV02 are annotated, but seperated into TEV02_01 and TEV02_02 in input_form.csv.")
    print("However, in plain_texts, TEV02 are two identical files. Since they are identical and they don't have a reveval border sentence, here we exclude them as well.\n")

    # Delete without annotations code and file
    for code in file_without_annotations.keys():
        del story_codes[code]

    # Delete file with no reval border
    for code in na_border:
        if code != "TEV02_02" and code != "TEV02_01":
            del story_codes[code]
    
    # In input_format.csv but not in data files
    file_without_text = [key for key in code_to_border.keys()  if key not in story_codes]
    print(f"These stories are not in {dir_path}", file_without_text, "\n")
    
    print("-----End Data Processing----- \n \n ")
    print("-----Start paritioning the story by reveal border sentence-----")

    code_to_partitions, code_to_plain_text, no_border_list = prep_partitions_and_plain_text(code_to_border, story_codes, dir_path)
    print("These cases really do not have reveal border, due to typo or other reasons", no_border_list, '\n')
    
    assert len(code_to_border) - len(file_without_text) - len(no_border_list) == len(code_to_partitions)
    print("The number of files is correct.")
    print("-----End partitioning and normalization-----\n")

    return code_to_partitions, code_to_plain_text