import pandas as pd

import random

import os



def semantic_mutate(text, rate):

    words = text.split()

    if len(words) < 5:

        return text

    

    num_mutations = max(1, int(len(words) * rate))

    new_words = list(words)

    

    for _ in range(num_mutations):

        mutation_type = random.choice(['swap', 'insert', 'delete', 'span_move'])

        

        if mutation_type == 'swap' and len(new_words) > 2:

            idx1, idx2 = random.sample(range(len(new_words)), 2)

            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

            

        elif mutation_type == 'insert':

            mutation_source = random.choice(words)

            insert_pos = random.randint(0, len(new_words))

            new_words.insert(insert_pos, mutation_source)

            

        elif mutation_type == 'delete' and len(new_words) > 8:

            del new_words[random.randint(0, len(new_words)-1)]

            

        elif mutation_type == 'span_move' and len(new_words) > 10:

            span_len = 2

            start_idx = random.randint(0, len(new_words) - span_len)

            span = new_words[start_idx : start_idx + span_len]

            del new_words[start_idx : start_idx + span_len]

            insert_pos = random.randint(0, len(new_words))

            new_words[insert_pos:insert_pos] = span



    return " ".join(new_words)



input_file = '/aul/homes/tnaya002/Desktop/lab/advpromptdetector/attacks/LeakAgent/dataset/attack_prompt.csv'

output_path = '/aul/homes/tnaya002/Desktop/lab/advpromptdetector/advGuard-main/ablation/3table1Diversity/attack/LeakAgent/Qwen2.5-7B-Instruct/prompt.csv'

leak_count = 1000



try:

    df_input = pd.read_csv(input_file)

    if 'text' not in df_input.columns:

        print(f"Error: Column 'text' not found in {input_file}")

        exit()

    current_data = df_input.to_dict('records')

except FileNotFoundError:

    print(f"Error: {input_file} not found.")

    exit()



# --- Processing ---

print(f"Expanding {len(current_data)} samples to {leak_count} using semantic mutations...")

expanded_data = []



while len(expanded_data) < leak_count:

    base_row = random.choice(current_data)

    

    rate = random.uniform(0.06, 0.15)

    

    mutated_text = semantic_mutate(base_row['text'], rate=rate)

    

    expanded_data.append({

        'idx': base_row.get('idx', 0), 

        'prompt': mutated_text,

    })



output_df = pd.DataFrame(expanded_data)



os.makedirs(os.path.dirname(output_path), exist_ok=True)



output_df.to_csv(output_path, index=False, encoding='utf-8')



print(f"Done! Successfully saved {leak_count} prompts to: {output_path}")


#import pandas as pd
#import random
#import re
#
#def prompt_data(text, rate=0.05):
#    words = text.split()
#    if not words:
#        return text
#    
#    num_to_mutate = max(1, int(len(words) * rate))
#    
#    new_words = list(words)
#    for _ in range(num_to_mutate):
#        mutation_source = random.choice(words)
#        insert_pos = random.randint(0, len(new_words))
#        new_words.insert(insert_pos, mutation_source)
#    
#    return " ".join(new_words)
#
#input_file = '/aul/homes/tnaya002/Desktop/lab/advpromptdetector/attacks/LeakAgent/dataset/attack_prompt.csv'  
#df = pd.DataFrame()
#
#try:
#    df = pd.read_csv(input_file)
#except FileNotFoundError:
#    print(f"Error: {input_file} not found.")
#    exit()
#
#leak_count = 1000
#current_data = df.to_dict('records')
#expanded_data = []
#
#print(f"Expanding {len(current_data)} samples to {leak_count}...")
#
#while len(expanded_data) < leak_count:
#    base_row = random.choice(current_data)
#    
#    rate = random.uniform(0.06, 0.19)
#    
#    leakAgent = prompt_data(base_row['text'], rate=rate)
#    
#    expanded_data.append({
#        'idx': base_row['idx'],
#        'prompt': leakAgent,
#    })
#
## 4. Save to CSV
#output_df = pd.DataFrame(expanded_data)
#output_df.to_csv('/aul/homes/tnaya002/Desktop/lab/advpromptdetector/advGuard-main/ablation/3table1Diversity/attack/LeakAgent/Qwen2.5-7B-Instruct/prompt.csv', index=False)
#
#print("Done! Saved to 'prompt.csv'")