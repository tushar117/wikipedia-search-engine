import sys
import os
import re
from datetime import datetime
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math

def fast_retrieval(offline_index_storage, file_name, offset_list, target_token, numeric=False):
    file_ptr = open(os.path.join(offline_index_storage, file_name), 'r', encoding='utf-8')
    low = 0
    high = len(offset_list) - 1
    result = ""
    while(low <= high):
        mid = int(low + ((high - low)/2))
        mid_offset = offset_list[mid]
        file_ptr.seek(mid_offset, 0)
        text = file_ptr.readline().strip()
        data = text.split(' ', 1)
        present_token = data[0]
        if numeric:
            present_token = float(present_token)
        if present_token == target_token:
            result = text
            break
        elif present_token < target_token:
            low = mid + 1
        else:
            high = mid - 1
    return result

def load_index_offset(index_folder, file_name, debug=False):
    offset_file_name = file_name+'-offset'
    offset_list = []
    if debug:
        print('[DEBUG] loading the offset list : %s' % offset_file_name)
    start_time = datetime.utcnow()
    with open(os.path.join(index_folder, offset_file_name), 'r') as offset_file:
        while True:
            line = offset_file.readline().strip()
            if not line or len(line)==0:
                break
            offset_list.append(int(line))
    time_delta = (datetime.utcnow() - start_time).total_seconds()
    if debug:
        print('[DEBUG] offset list : %s is loaded in %.2f seconds' % (offset_file_name, time_delta))
    return offset_list

def load_primary_index_offset(index_folder, offset_file, debug=False):
    offset_file_path = os.path.join(os.path.abspath(index_folder), offset_file)
    primary_index_offset = []
    if debug:
        print("[DEBUG] loading primary index offset data from file -- %s " % offset_file_path)
    start_time = datetime.utcnow()
    with open(offset_file_path, 'r', encoding='utf-8') as txt_file:
        for line in txt_file.readlines():
            data = line.strip().split()
            primary_index_offset.append([data[0], data[1], data[2]])
    time_delta = (datetime.utcnow() - start_time).total_seconds()
    if debug:
        print("[DEBUG] loaded primary index offset in %4.2f seconds" % (time_delta))
    return primary_index_offset

def load_primary_index(index_folder, f_index_name, target_tokens, debug=False):
    index_dump_file = os.path.join(os.path.abspath(index_folder), f_index_name)
    
    inverted_index = {}
    index_offset = load_index_offset(index_folder, f_index_name, debug=debug)
    if debug:
        print("[DEBUG] loading targeted inverted index from file -- %s " % index_dump_file)
    start_time = datetime.utcnow()
    for token in target_tokens:
        result = fast_retrieval(index_folder, f_index_name, index_offset, token).strip()
        entries = result.split()
        if len(entries) < 3:
            if debug:
                print("[ERROR] no posting entry found for token : %s in index file : %s" % (token, f_index_name))
            continue
        #initialize the entry in inverted list
        inverted_index[entries[0]] = {
            'total_frequency': int(entries[1]),
            'posting_list': [],
        }
        count = 0
        for posting_data in entries[2:]:
            if count % 2 == 0:
                posting_entry = [posting_data]
            else:
                posting_entry.append(posting_data)
                inverted_index[entries[0]]['posting_list'].append(posting_entry)
            count +=1
    time_delta = (datetime.utcnow() - start_time).total_seconds()
    if debug:
        print("[DEBUG] loaded targeted inverted index in %4.2f seconds" % (time_delta))
    return inverted_index

def search_within_indexes(stemmer, stop_words, query, inv_index, title_index, threshold=10):
    #find the category search
    category_regex = re.compile('\s*\w+:(\w+)\s*')
    relevant_words = re.findall(category_regex, query)
    #remove categorical searcch
    search_query = re.sub(category_regex, '', query)
    #add back the relevant terms
    search_query += ' '.join(relevant_words)
    #print("search query : %s" % search_query)
    #word tokenization
    search_query = word_tokenize(search_query)
    #stopword removal
    search_query = list(set(search_query).difference(set(stop_words)))
    #stemming of words and case folding
    search_query = [stemmer.stem(w.lower()) for w in search_query]
    #print("final search query : %s" % search_query)
    search_ids = []
    for token in search_query:
        intermediate_ids = inv_index.get(token, '')
        intermediate_ids = [x[0] for x in intermediate_ids]
        search_ids = list(set(search_ids).union(set(intermediate_ids)))
    search_results = []
    for doc_id in search_ids[:threshold]:
        title = title_index.get(doc_id, None)
        if title is not None:
            search_results.append(title)
    return search_results

def locate_primary_index_files(word_desc, primary_index_offset, debug=True):
    index_hits = {}
    for word_info in word_desc:
        word = word_info[0]
        low = 0
        high = len(primary_index_offset)
        hit = False
        while(low <= high):
            mid = low + int((high - low)/2)
            #bucket found
            if word == primary_index_offset[mid][1] or word == primary_index_offset[mid][2] or (primary_index_offset[mid][1] < word and word < primary_index_offset[mid][2]): 
                index_bucket = index_hits.get(primary_index_offset[mid][0], None)
                # create new entry
                if index_bucket is None:
                    index_hits[primary_index_offset[mid][0]] = [word_info]
                else:
                    index_hits[primary_index_offset[mid][0]].append(word_info)
                hit = True
                break
            elif word < primary_index_offset[mid][1]:
                high = mid-1
            else:
                low = mid+1
        if not hit and debug:
            print("[DEBUG] no index file found for word : %s" % word)
    if debug:
        print("[DEBUG] intermediate primary index search : %s" % index_hits)
    hit_results = []
    for key, value in sorted(index_hits.items(), key=lambda kv: len(kv[1]), reverse=True):
        hit_results.append([key, value])
    return hit_results

def calculate_rank(path_to_index_folder, file_locations, results, total_documents, threshold=10, debug=False):
    for entry in file_locations:
        index_file = entry[0]
        word_desc = entry[1]
        target_tokens = [word[0] for word in word_desc]
        inverted_index = load_primary_index(path_to_index_folder, index_file, target_tokens, debug=debug)
        for word in word_desc:
            token = word[0]
            field_activation = word[1]
            index_entry = inverted_index.get(token, None)
            if index_entry is not None:
                calculate_weight(index_entry, results, field_activation, total_documents)
    final_result_id = []
    count = 0
    for key, value in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
        if count >=threshold:
            break
        final_result_id.append(key)
        count += 1
    return final_result_id
    

def calculate_weight(index_entry, results, field_activation, total_documents):
    total_frequency = float(index_entry['total_frequency'])
    posting_list = index_entry['posting_list']
    total_docs = float(len(posting_list))
    for entry in posting_list:
        doc_id = entry[0]
        field = entry[1]
        weight = calculate_field_weight(field_activation, field) * math.log(total_documents/float(total_docs))
        
        result_entry = results.get(doc_id, None)
        if result_entry is None:
            results[doc_id] = weight
        else:
            results[doc_id] += weight 

def calculate_field_weight(field_activation, entry_field):
    result = 0.0
    factor = {
        't': 0.25,
        'b': 0.25,
        'i': 0.20,
        'c': 0.1,
        'l': 0.1,
    }
    entry_field = entry_field.strip()
    for field in field_activation:
        if field in entry_field:
            index = entry_field.find(field)
            weight = extract_number(index+1, entry_field)
            result += float(factor[field] * (1+math.log(float(weight)))) 
    return result

def extract_number(index, word):
    res = 0.0
    while(index < len(word)):
        if '0'<=word[index] and word[index]<='9':
            res = res*10 + float(ord(word[index]) - ord('0'))
            index+=1
            continue
        break
    return res+1e-3

def print_results_from_title_map(results, offset_list, index_folder, title_map_name, debug=False):
    count = 0
    print("Results >> ")
    for doc_id in results:
        data = fast_retrieval(index_folder, title_map_name, offset_list, int(doc_id), numeric=True).strip()
        if len(data) > 0:
            doc_title = data.split(' ', 1)[1] 
            count += 1
            if debug:
                print(">>%s %s" %(doc_id, doc_title))
            else:
                print(">>%s" % (doc_title))
        else:
            if debug:
                print(">> !!! error encountered for %d" % doc_id)
    if count == 0 :
        print(">> No results found !!!")

def search(path_to_index_folder, result_count=10):
    search_results = []
    title_map_name = "doc_title_map"
    
    primary_index_offset_file = "primary_index_file_offset"
    debugMode = True
    # load utilities
    stop_words = stopwords.words("english")
    stemmer = PorterStemmer()
    #load indexes
    title_map_offset = load_index_offset(path_to_index_folder, title_map_name, debug=debugMode)
    total_document_count = len(title_map_offset) 
    
    primary_index_offset = load_primary_index_offset(path_to_index_folder, primary_index_offset_file, debug=debugMode)
    print('Search across %s pages ...' % (total_document_count))
    while True:
        query = input('\nType in your query:\n')
        start = datetime.utcnow()
        # query = query.lower()
        #find the category search
        category_regex = re.compile('\s*\w+:(\w+)\s*')
        relevant_words = re.findall(category_regex, query)
        final_search_query = []
        #field queries
        if len(relevant_words) > 0:
            group_field = {}
            print("working with field query search :")
            global_words = re.findall(r'[t|i|c|l|b]:([^:]*)(?!\S)', query)
            global_fields = re.findall(r'([t|i|c|l|b]):', query)
            for index, word in enumerate(global_words):
                #word tokenization
                search_query = word_tokenize(word)
                #stopword removal
                search_query = list(set(search_query).difference(set(stop_words)))
                #stemming of words and case folding
                search_query = [stemmer.stem(w.lower()) for w in search_query]
                for query_token in search_query:
                    isPresent = group_field.get(query_token, None)
                    if isPresent is not None:
                        group_field[query_token] += global_fields[index]
                    else:
                        group_field[query_token] = global_fields[index]

            for key, value in group_field.items():
                final_search_query.append([key, value])
        else:
            #word tokenization
            search_query = word_tokenize(query)
            #stopword removal
            search_query = list(set(search_query).difference(set(stop_words)))
            #stemming of words and case folding
            search_query = [stemmer.stem(w.lower()) for w in search_query]
            for query_token in search_query:
                final_search_query.append([query_token, 'ticlb'])
        file_locations = locate_primary_index_files(final_search_query, primary_index_offset, debug=debugMode)
        results = {}
        final_results = calculate_rank(path_to_index_folder, file_locations, results, total_document_count, threshold=10, debug=debugMode)
        print_results_from_title_map(final_results, title_map_offset, path_to_index_folder, title_map_name, debug=debugMode)
        time_elapsed = (datetime.utcnow() - start).total_seconds()
        print("[INFO] : search completed in %.2f seconds" % time_elapsed)

def main():
    if len(sys.argv) != 2:
      print("error : invalid number of argument passed [1] arguments required, need path to index_folder")
      return -1
    path_to_index = sys.argv[1]
    search(path_to_index)

if __name__ == '__main__':
    main()
