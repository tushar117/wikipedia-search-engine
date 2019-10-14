import xml.sax
from datetime import datetime
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# from functools import reduce
import heapq
from collections import defaultdict

import numpy as np
import sys
import os
import shutil
import multiprocessing


class WikiPageHandler(xml.sax.ContentHandler):
   def __init__(self, process_configuration, process_stat):
      self.allowed_tags = ['title', 'id', 'text']
      self.present_tag = ""
      self.data = {}
      self.enable_read = False
      self.activate_read = False
      self.doc_id = 1
      #read the configuration
      #process
      self.process_id = process_configuration['process_id']
      self.offset = process_configuration['offset']
      self.debug_mode = process_configuration['debug_mode']
      #index
      self.offline_block_size = process_configuration['offline_block_size']
      self.offline_index_counter = self.process_id
      #storage
      self.doc_parse_count = 0
      self.secondary_index_file_name = process_configuration['inverted_index_file']
      self.title_map_name = process_configuration['doc_title_map']
      self.offline_index_storage = process_configuration['offline_index_storage']
      #nlp & RE
      self.wiki_pattern_matching = process_configuration['wiki_pattern_matching']
      self.stemmer = process_configuration['stemmer']
      self.stop_words = process_configuration['stop_words']
      #stats
      self.process_statistics = process_stat
      self.process_statistics['index_start_time'] = datetime.utcnow()
      self.process_statistics['start_index'] = self.process_id
      self.process_statistics['end_index'] = self.process_id
      self.process_statistics['inverted_index'] = {}
      self.process_statistics['doc_id_title_hash'] = {}

   # Call when an element starts
   def startElement(self, tag, attributes):
      self.present_tag = tag
      self.activate_read = True
      if tag == "page":
         self.data = {}
         for tag_name in self.allowed_tags:
            self.data[tag_name] = []
         self.enable_read = False
         if (self.doc_id - self.process_id) % self.offset == 0:
            self.enable_read = True

   # Call when an elements ends
   def endElement(self, tag):
      self.activate_read = False
      if tag == "page":
         self.enable_read = False
         #assemble the string
         if (self.doc_id - self.process_id) % self.offset == 0:
            for tag_name in self.allowed_tags:
               result = str(''.join(self.data.get(tag_name, ''))).strip()
               if tag_name == 'text':
                  info, category, links, body = self.post_processing(result, self.data['id'], debugMode=self.debug_mode)
                  title_tokens = self.final_text_processing(self.data['title'])
                  self.populate_inverted_index(self.doc_id, title_tokens, info, category, links, body)
               self.data[tag_name] = result
            self.process_statistics['doc_id_title_hash'][self.doc_id] = self.data['title']
            self.doc_parse_count += 1

         #store the indexes offline to save memory
         if self.doc_parse_count % self.offline_block_size == 0 and self.doc_parse_count!=0:
            time_elapsed = (datetime.utcnow() - self.process_statistics['index_start_time']).total_seconds()
            print(">process_id[%s] : [%s] processed the block with %d words in %.2f seconds" % (self.process_id, self.offline_index_counter, len(self.process_statistics['inverted_index']), time_elapsed))
            store_partial_index_offline(self.process_id, self.process_statistics['inverted_index'], self.offline_index_storage, self.secondary_index_file_name, self.offline_index_counter)
            store_partial_title_map_offline(self.process_id, self.offline_index_counter, self.process_statistics['doc_id_title_hash'], self.offline_index_storage, self.title_map_name)
            self.process_statistics['inverted_index'] = {}
            self.process_statistics['doc_id_title_hash'] = {}
            self.process_statistics['end_index'] = self.offline_index_counter
            self.offline_index_counter += self.offset
            self.doc_parse_count = 0
            self.process_statistics['index_start_time'] = datetime.utcnow()
         self.doc_id += 1
   
   # Call when a character is read
   def characters(self, content):
      if self.present_tag != "page" and (self.enable_read and self.activate_read) and self.present_tag in self.allowed_tags:
         result = content.strip()
         if self.present_tag == 'id' and len(self.data['id']) != 0:
            result = ''
         if len(result) > 0:
            self.data[self.present_tag].append(result)
   
   #this will populate the ide
   def populate_inverted_index(self, doc_id, title, info, category, link, body):
      combined_words = defaultdict(int)
      field_count_manager = []

      title_map = defaultdict(int)
      field_count_manager.append(title_map)
      for word in title:
         combined_words[word] += 1
         title_map[word] += 1
      
      info_map = defaultdict(int)
      field_count_manager.append(info_map)
      for word in info:
         combined_words[word] += 1
         info_map[word] += 1
      
      category_map = defaultdict(int)
      field_count_manager.append(category_map)
      for word in category:
         combined_words[word] += 1
         category_map[word] += 1

      link_map = defaultdict(int)
      field_count_manager.append(link_map)
      for word in link:
         combined_words[word] += 1
         link_map[word] += 1

      body_map = defaultdict(int)
      field_count_manager.append(body_map)
      for word in body:
         combined_words[word] += 1
         body_map[word] += 1

      field_alias = ['t', 'i', 'c', 'l', 'b']

      for word, total_freq in combined_words.items():
         index_string = ''
         for field_index in range(len(field_alias)):
            field_count = field_count_manager[field_index][word]
            if field_count != 0:
               index_string += field_alias[field_index]+str(field_count)
         total_word_count = total_freq

         isPresent = self.process_statistics['inverted_index'].get(word, None)
         #create a new entry if entry is not present
         if isPresent is None:
            self.process_statistics['inverted_index'][word] = {'total_count': 0, 'posting_list': []}
         self.process_statistics['inverted_index'][word]['total_count'] += total_word_count
         self.process_statistics['inverted_index'][word]['posting_list'].append([doc_id, index_string])

   #postprocessing of the text returned after the parsing
   def post_processing(self, content, wiki_id, debugMode=False):
      
      page_info = []
      page_category = []
      page_links = []
      page_body = []

      #setting up the page information
      result, content = self.extract_information(content, wiki_id, debugMode=debugMode)
      page_info.append(result)
      result, content = self.extract_infobox(content, wiki_id, debugMode=debugMode)
      page_info.append(result)
      #setting up page category
      result, content = self.extract_category(content, wiki_id, debugMode=debugMode)
      page_category.append(result)
      #setting up the page links
      result, content = self.extract_wiki_links(content, wiki_id, debugMode=debugMode)
      page_links.append(result)
      # insert the body in page_body
      page_body.append(self.get_content_body(content, wiki_id, debugMode=debugMode))
      return self.final_text_processing(page_info), self.final_text_processing(page_category), self.final_text_processing(page_links), self.final_text_processing(page_body)

   # just clean the extra data 
   def get_content_body(self, wiki_content, wiki_id, debugMode=False):
      regex = re.compile(self.wiki_pattern_matching["comments"])
      new_content = re.sub(regex, ' ', wiki_content)
      regex = re.compile(self.wiki_pattern_matching["styles"])
      new_content = re.sub(regex, ' ', new_content)
      regex = re.compile(self.wiki_pattern_matching["references"])
      new_content = re.sub(regex, ' ', new_content)
      new_content = self.remove_all_tags(new_content)
      regex = re.compile(self.wiki_pattern_matching["curly_braces"])
      new_content = re.sub(regex, ' ', new_content)
      regex = re.compile(self.wiki_pattern_matching["square_braces"])
      new_content = re.sub(regex, ' ', new_content)
      #remove links
      new_content = self.remove_all_urls(new_content)
      #remove footers
      new_content = self.strip_footers(new_content)
      return new_content

   def strip_footers(self, content):
      labels = ["References", "Further reading", "See also", "Notes"]
      for l in labels:
         regex = "==%s==" % (labels)
         found = re.search(regex, content)
         if found is not None:
            # get the index of the search
            content = content[0:found.start()-1]
      return content 

   #only extract infobox from the wiki page
   def extract_infobox(self, wiki_content, wiki_id, debugMode=False):
      start_index = wiki_content.find(self.wiki_pattern_matching["infobox"])
      result_string = ''
      if start_index < 0:
         return result_string, wiki_content
      end = len(wiki_content)
      bracket_match_count = 2
      start_index += len(self.wiki_pattern_matching["infobox"])
      index = start_index
      while(index < end):
         if wiki_content[index] == '}':
            bracket_match_count -= 1
         if wiki_content[index] == '{':
            bracket_match_count += 1
         if bracket_match_count == 0:
            break
         index += 1
      if bracket_match_count !=0 and debugMode:
         print("extract_infobox_warning : malformed infobox found for wiki id %s" % (wiki_id))
         return result_string, wiki_content
      result_string = wiki_content[start_index:index-1]
      result_string = self.remove_all_urls(result_string)
      new_wiki_content = wiki_content[0:start_index-1] + wiki_content[index+1:-1]
      return self.remove_all_tags(self.filter_content(result_string)), new_wiki_content
   
   #only extract information from the wiki page
   def extract_information(self, wiki_content, wiki_id, debugMode=False):
      start_index = wiki_content.find(self.wiki_pattern_matching["information"])
      result_string = ''
      if start_index < 0:
         return result_string, wiki_content
      end = len(wiki_content)
      bracket_match_count = 2
      start_index += len(self.wiki_pattern_matching["information"])
      index = start_index
      while(index < end):
         if wiki_content[index] == '}':
            bracket_match_count -= 1
         if wiki_content[index] == '{':
            bracket_match_count += 1
         if bracket_match_count == 0:
            break
         index += 1
      if bracket_match_count !=0 and debugMode:
         print("extract_information_warning : malformed information found for wiki id %s" % (wiki_id))
         return result_string, wiki_content
      result_string = wiki_content[start_index:index-1]
      result_string = self.remove_all_urls(result_string)
      new_wiki_content = wiki_content[0:start_index-1] + wiki_content[index+1:-1]
      return self.remove_all_tags(self.filter_content(result_string)), new_wiki_content

   #only extract the category body
   def extract_category(self, wiki_content, wiki_id, debugMode=False):
      category_regex = re.compile(self.wiki_pattern_matching["category"], re.IGNORECASE)
      result_string = re.findall(category_regex, wiki_content)
      result_string = ''.join(result_string)
      result_string = self.remove_all_urls(result_string)
      new_wiki_content = re.sub(category_regex, '', wiki_content)
      return self.remove_all_tags(self.filter_content(result_string)), new_wiki_content

   def extract_wiki_links(self, wiki_content, wiki_id, debugMode=False):
      wiki_links_regex = re.compile(self.wiki_pattern_matching["wiki_links"], re.IGNORECASE)
      temp_string = re.findall(wiki_links_regex, wiki_content)
      result_string = []
      for value in temp_string:
         if ':' not in value:
            result_string.append(value)
      result_string = ''.join(result_string)
      result_string = self.remove_all_urls(result_string)
      new_wiki_content = re.sub(wiki_links_regex, '', wiki_content)
      return self.remove_all_tags(self.filter_content(result_string)), new_wiki_content

   def remove_all_tags(self, wiki_content):
      tag_regex = re.compile("<.*?>")
      return re.sub(tag_regex, '', wiki_content)

   def remove_all_urls(self, wiki_content):
      #remove links
      regex = re.compile(self.wiki_pattern_matching["url"])
      wiki_content = re.sub(regex, ' ', wiki_content)
      regex = re.compile(self.wiki_pattern_matching["www"])
      wiki_content = re.sub(regex, ' ', wiki_content)
      return wiki_content

   # it process the text_list to execute case_folding, tokenization, stop_word removal
   # and stemming in order
   def final_text_processing(self, text_list):
      result = self.filter_content(''.join(text_list)).lower()
      #word tokenization
      result = word_tokenize(result)
      # result = result.split()
      #stopword removal
      result = list(set(result).difference(set(self.stop_words)))
      #stemming of words and case folding
      result = [self.stemmer.stem(w) for w in result]
      return result

   def filter_content(self, content):
      filters = set(['(', '{', '[', ']', '}', ')', '=', '|', '?', ',', '+', '\'', '\\', '*', '#', ';', '!', '\"', '%'])
      content = content.strip()
      if len(content) == 0:
         return content
      if len(set(content).intersection(filters)) == 0:
         return content
      for elem in filters:
         content = content.replace(elem, ' ')
      return content

def store_partial_index_offline(process_id, inverted_index, offline_index_storage, index_file_name, file_index_id):
   offline_index_file = index_file_name + '-' + str(file_index_id)
   index_dump_file = os.path.join(offline_index_storage, offline_index_file)
   #print("writing inverted index to file -- %s " % index_dump_file)
   start_time = datetime.utcnow()
   with open(index_dump_file, 'w+', encoding='utf-8') as txt_file:
      for word in sorted(inverted_index.keys()):
         posting_list = inverted_index[word] 
         total_freq_count = posting_list['total_count']
         txt_file.write("%s %s" % (word, total_freq_count))
         for entry in posting_list['posting_list']:
            #value = entry[0] - prev
            txt_file.write(" %s %s" %(entry[0], entry[1]))
            #prev = entry[0]
         txt_file.write("\n")
   time_delta = (datetime.utcnow() - start_time).total_seconds()
   file_size = os.path.getsize(index_dump_file)/float(1<<20)
   print(">process_id[%s] : [%s] saved index block on disk with name : %s in %.2f seconds -- [%.2f MB]" % (process_id, file_index_id, offline_index_file, time_delta, file_size))
   return file_size

def store_partial_title_map_offline(process_id, block_count, doc_id_title_hash, offline_index_storage, title_map_name):
   #creating doc title map file
   title_map_file_name = title_map_name+'-'+str(process_id)
   title_map_file = os.path.join(offline_index_storage, title_map_file_name)
   start_time = datetime.utcnow()
   with open(title_map_file, 'a+', encoding='utf-8') as txt_file:
      for doc_id, title in doc_id_title_hash.items():
         txt_file.write("%d %s\n" % (doc_id, title))
   time_delta = (datetime.utcnow() - start_time).total_seconds()
   print(">process_id[%s] : [%s] title_map_hash written to the disk in %.2f seconds" % (process_id, block_count, time_delta))
   print("----"*25)

def merge_title_map_files(cfg):
   merged_title_file = os.path.join(cfg['offline_index_storage'], cfg['doc_title_map'])
   offset_file_name = os.path.join(cfg['offline_index_storage'], cfg['doc_title_map']+'-offset')
   
   delete_temporary_index_files(cfg['offline_index_storage'], [cfg['doc_title_map'], cfg['doc_title_map']+'-offset'], verbose=True)
   merged_title_offset_file = open(offset_file_name, 'a+')
   title_counter = 0
   present_offset = 0
   print('[INFO] : merging the partial title map files..')
   start_time = datetime.utcnow()
   with open(merged_title_file, 'a+', encoding='utf-8') as txt_file:
      temp_file_handler = []
      temp_file_open_status = []
      for index in range(1, cfg['offset']+1):
         temp_file_name = cfg['doc_title_map']+'-'+str(index)
         temp_file_handler.append(open(os.path.join(cfg['offline_index_storage'], temp_file_name), 'r', encoding='utf-8'))
         temp_file_open_status.append(True)
      #merge the title files
      while any(temp_file_open_status):
         file_index = title_counter % cfg['offset']
         temp_entry = temp_file_handler[file_index].readline().strip()
         #close the empty file
         title_counter += 1
         if not temp_file_open_status[file_index] or not temp_entry or len(temp_entry)==0:
            temp_file_open_status[file_index] = False
            temp_file_handler[file_index].close()
            continue
         final_content = '%s\n' % temp_entry
         txt_file.write(final_content)
         merged_title_offset_file.write('%d\n' % present_offset)
         present_offset += (len(final_content.strip().encode())+len(os.linesep))
   #finally close the merged_offset_file
   merged_title_offset_file.close()
   time_delta = (datetime.utcnow() - start_time).total_seconds()
   print('[INFO] : merged title map files in %.2f seconds, total titles : %s' % (time_delta, title_counter))
   return title_counter

def merge_all_index_files(cfg, delete_temp_files=False):
   offline_index_counter = cfg['offline_index_counter']
   stats_file_name = cfg['primary_stats_file_name']
   index_file_name = cfg['inverted_index_file']
   offline_index_storage = cfg['offline_index_storage']
   primary_index_file_name = cfg['primary_index_file_name']
   primary_index_offset_name = cfg['primary_index_offset_name']
   primary_index_size = cfg['primary_index_size']
   total_primary_index_size = 0

   #clean the primary index files and primary index offset
   delete_files = []
   for target_file_name in os.listdir(offline_index_storage):
      if stats_file_name in target_file_name or primary_index_file_name in target_file_name or primary_index_offset_name in target_file_name:
         delete_files.append(target_file_name)
   #actually deletes the files
   delete_temporary_index_files(offline_index_storage, delete_files, verbose=True)

   index_file_pointer = [None]*offline_index_counter
   primary_heap = []
   present_token = ['']*offline_index_counter
   present_freq = [0]*offline_index_counter
   present_content = ['']*offline_index_counter
   present_token_count = [0]*offline_index_counter
   index_file_open_status = [False]*offline_index_counter
   primary_index_offset = []

   delete_files = []
   for i in range(offline_index_counter):
      offline_index_file = index_file_name + '-' + str(i+1) 
      index_dump_file = os.path.join(offline_index_storage, offline_index_file)
      index_file_pointer[i] = open(index_dump_file, 'r', encoding='utf-8')
      temp_entry = index_file_pointer[i].readline().strip()
      delete_files.append(index_file_name+'-'+str(i+1))
      #close the empty file
      if not temp_entry or len(temp_entry)==0:
         # print(">> closing the file : %s after processing %s words" % ((index_file_name+'-'+str(i), present_token_count[i])))
         index_file_pointer[i].close()
         continue
      temp_entry = temp_entry.split(' ', 2)
      if len(temp_entry) != 3:
         print(">> wrongly formatted index entries for index file %s : %s" % ((index_file_name+'-'+str(i+1)), temp_entry))
         continue
      #initializing the head pointer values for each index files
      present_token[i] = temp_entry[0].strip()
      present_freq[i] = int(temp_entry[1].strip())
      present_content[i] = temp_entry[2].strip()
      present_token_count[i] += 1
      #setting file open status
      index_file_open_status[i] = True

      if present_token[i] not in primary_heap:
         heapq.heappush(primary_heap, present_token[i])

   total_unique_tokens = 0
   primary_index_counter = 0

   final_index_file_name = primary_index_file_name + '-' + str(primary_index_counter)
   final_index_offset_file_name = final_index_file_name + '-offset'
   primary_index_file = open(os.path.join(offline_index_storage, final_index_file_name), 'w+', encoding='utf-8')
   primary_index_offset_file = open(os.path.join(offline_index_storage, final_index_offset_file_name), 'w+', encoding='utf-8')
   start_time = datetime.utcnow()
   start_token = 0
   prev_token = 0
   present_offset = 0
   start_flag = True

   while(any(index_file_open_status)):
      target_word = heapq.heappop(primary_heap)
      total_unique_tokens +=1
      if total_unique_tokens % primary_index_size == 0:
         time_elapsed = (datetime.utcnow() - start_time).total_seconds()
         primary_index_file.close()
         primary_index_offset_file.close()
         present_offset = 0
         primary_index_offset.append([final_index_file_name, start_token, prev_token])
         file_size = os.path.getsize(os.path.join(offline_index_storage, final_index_file_name))/float(1<<20)
         total_primary_index_size += file_size
         print(">> [primary index : %s] saved primary index block on disk with name : %s in %.2f seconds -- [%.2f MB]" % (primary_index_counter, final_index_file_name, time_elapsed, file_size))
         print("===="*32)
         primary_index_counter += 1
         final_index_file_name = primary_index_file_name + '-' + str(primary_index_counter)
         final_index_offset_file_name = final_index_file_name + '-offset'
         primary_index_file = open(os.path.join(offline_index_storage, final_index_file_name), 'w+', encoding='utf-8')
         primary_index_offset_file = open(os.path.join(offline_index_storage, final_index_offset_file_name), 'w+', encoding='utf-8')
         start_time = datetime.utcnow()
         start_flag = True
      #only to mark the starting token of the index block
      if start_flag:
         start_token = target_word
         start_flag = False
      
      target_freq = 0
      target_content = []
      for i in range(offline_index_counter):
         if not index_file_open_status[i]:
            continue
         #it's a match
         if present_token[i] == target_word:
            target_freq+=present_freq[i]
            target_content.append(present_content[i])
            #now update the read pointer
            temp_entry = index_file_pointer[i].readline().strip()
            #close the empty file
            if not temp_entry or len(temp_entry)==0:
               # print(">> closing the file : %s after processing %s words" % ((index_file_name+'-'+str(i)), present_token_count[i]))
               index_file_open_status[i] = False
               index_file_pointer[i].close()
               continue
            temp_entry = temp_entry.split(' ', 2)
            #close wrongly formatted file
            if len(temp_entry) != 3:
               print(">> wrongly formatted index entries for index file %s : %s. closing it" % ((index_file_name+'-'+str(i+1)), temp_entry))
               index_file_open_status[i] = False
               index_file_pointer[i].close()
               continue 
            #everything is fine, now update the head pointer values for each index files
            present_token[i] = temp_entry[0].strip()
            present_freq[i] = int(temp_entry[1].strip())
            present_content[i] = temp_entry[2].strip()
            present_token_count[i] += 1
            if present_token[i] not in primary_heap:
               heapq.heappush(primary_heap, present_token[i])
      final_write_content = "%s %s %s\n" %(target_word, target_freq, ' '.join(target_content))      
      primary_index_file.write(final_write_content)
      primary_index_offset_file.write('%d\n'% present_offset)
      present_offset += (len(final_write_content.strip().encode())+len(os.linesep))
      prev_token = target_word
   #closes the last block 
   if total_unique_tokens % primary_index_size !=0:
      time_elapsed = (datetime.utcnow() - start_time).total_seconds()
      primary_index_file.close()
      primary_index_offset_file.close()
      file_size = os.path.getsize(os.path.join(offline_index_storage, final_index_file_name))/float(1<<20)
      total_primary_index_size += file_size
      print(">> [primary index : %s] saved primary index block on disk with name : %s in %.2f seconds -- [%.2f MB]" % (primary_index_counter, final_index_file_name, time_elapsed, file_size))
      print("===="*32)
      primary_index_offset.append([final_index_file_name, start_token, prev_token])
      primary_index_counter += 1
   
   #populate the primary index offset
   offset_file = os.path.join(offline_index_storage, primary_index_offset_name)
   with open(offset_file, 'w+', encoding='utf-8') as o_file:
      for data in primary_index_offset:
         o_file.write("%s %s %s\n" % (data[0], data[1], data[2]))
   
   #finally delete the secondary index files
   if delete_temp_files:
      delete_temporary_index_files(offline_index_storage, delete_files, verbose=True)

   return total_unique_tokens, primary_index_counter, total_primary_index_size              

def delete_temporary_index_files(offline_index_storage, file_list, verbose=False):
   success = True
   for index_file_name in file_list:
      index_dump_file = os.path.join(offline_index_storage, index_file_name)
      #check if exists
      if not os.path.isfile(index_dump_file):
         continue
      try:
         os.remove(index_dump_file)
      except Exception as e:
         if verbose:
            print("[ERROR] unable to remove file : %s, error: %s" % (index_dump_file, str(e)))
         success = success & False
   if len(file_list)>0 and success and verbose:
      print("[INFO] successfully deleted the local file : %s" % (file_list))

def create_secondary_index(xml_dump_file, process_configuration, process_stats):
   # create an XMLReader
   parser = xml.sax.make_parser()
   # turn off namepsaces
   parser.setFeature(xml.sax.handler.feature_namespaces, 0)

   # override the default ContextHandler
   Handler = WikiPageHandler(process_configuration, process_stats)
   parser.setContentHandler(Handler)
   parser.parse(xml_dump_file)
   #writing the last block
   if len(process_stats['inverted_index']) > 0:
      time_elapsed = (datetime.utcnow() - process_stats['index_start_time']).total_seconds()
      process_stats['end_index'] += process_configuration['offset']
      print(">process_id[%s] : [%s] processed the block with %d words in %.2f seconds" % (process_configuration['process_id'], process_stats['end_index'], len(process_stats['inverted_index']), time_elapsed))
      store_partial_index_offline(process_configuration['process_id'], process_stats['inverted_index'], process_configuration['offline_index_storage'], process_configuration['inverted_index_file'], process_stats['end_index'])
      store_partial_title_map_offline(process_configuration['process_id'], process_stats['end_index'], process_stats['doc_id_title_hash'], process_configuration['offline_index_storage'], process_configuration['doc_title_map'])
      process_stats['inverted_index'] = {}
      process_stats['doc_id_title_hash'] = {}
   
   #store the statistics
   total_index_bytes = 0
   for file_index in range(process_stats['start_index'], process_stats['end_index']+process_configuration['offset'], process_configuration['offset']):
      file_name =  process_configuration['inverted_index_file']+'-'+str(file_index)
      file_size = os.path.getsize(os.path.join(process_configuration['offline_index_storage'], file_name))
      total_index_bytes += file_size
   stats_file_name = process_configuration['stats_file_name']+'-'+str(process_configuration['process_id'])
   with open(os.path.join(process_configuration['offline_index_storage'], stats_file_name), 'w+') as stats_file:
      stats_file.write("%d %d %d\n" % (process_stats['start_index'], process_stats['end_index'], total_index_bytes))

def get_configuration(offline_index_storage):
   config = {
      'offset':2,
      'debug_mode': False,
      'offline_block_size': 10000,
      'inverted_index_file': 'inverted_index_file',
      'doc_title_map': 'doc_title_map',
      'offline_index_storage': offline_index_storage,
      'wiki_pattern_matching': {
         "information": "{{information",
         "infobox" : "{{Infobox",
         "category" : "\[\[Category:\s*(.*?)\]\]",
         "wiki_links": "\[\[(.*?)\]\]",
         "comments" : "<--.*?-->",
         "styles" : "\[\|.*?\|\]",
         "curly_braces": "{{.*?}}",
         "square_braces": "\[\[.*?\]\]",
         "references": "<ref>.*?</ref>",
         "url": "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
         "www": "www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
      },
      'stemmer' : PorterStemmer(), 
      'stop_words' : stopwords.words("english"),
      'stats_file_name' : 'process-stats',
      'primary_index_file_name' : 'prime_index_file',
      'primary_index_offset_name' : 'primary_index_file_offset',
      'primary_index_size': 100000,
      'primary_stats_file_name': 'merge_stats',
   }
   return config

def display_config(cfg, allowed_fields):
   for key in cfg.keys():
      if key in allowed_fields:
         print('    %s : %s' % (key, cfg[key]))

def clean_index_directory(offline_index_storage):
   for the_file in os.listdir(offline_index_storage):
      file_path = os.path.join(offline_index_storage, the_file)
      try:
         if os.path.isfile(file_path):
            os.unlink(file_path)
         elif os.path.isdir(file_path): shutil.rmtree(file_path)
      except Exception as e:
         print("unable to delete path %s. error : %s" % (the_file, str(e)))

def build_secondary_index(xml_dump_file, offline_index_storage, empty_dir=True):
   index_start_time = datetime.utcnow()
   cfg = get_configuration(offline_index_storage)
   display_cfg_fields = ['offset', 'debug_mode', 'offline_block_size', 'inverted_index_file', 'doc_title_map', 'offline_index_storage', 'stats_file_name']
   print("[INFO] creating inverted index using %s , please wait..." % xml_dump_file)
   if empty_dir:
      print("[INIT] cleaning the index directory.")
      clean_index_directory(offline_index_storage)
      print("[INFO] directory cleaned")
   #clean the primary index files and primary index offset
   delete_files = []
   for target_file_name in os.listdir(offline_index_storage):
      if cfg['inverted_index_file'] in target_file_name or cfg['doc_title_map'] in target_file_name or cfg['stats_file_name'] in target_file_name:
         delete_files.append(target_file_name)
   #actually deletes the files
   delete_temporary_index_files(offline_index_storage, delete_files, verbose=True)
   
   print("[CONFIG] secondary index creation running with configuration :")
   display_config(cfg, display_cfg_fields)
   print('\n[CRTICAL] spawning %d processes for secondary index creation.' % (cfg['offset']))
   process_handlers = []
   for process_id in range(1, cfg['offset']+1):
      cfg = get_configuration(offline_index_storage)
      cfg['process_id'] = process_id
      prc = multiprocessing.Process(target=create_secondary_index, args=(xml_dump_file, cfg, {}))
      process_handlers.append(prc)
   #execute the process
   for prc in process_handlers:
      prc.start()
   #wait for the process to complete
   for prc in process_handlers:
      prc.join()
   index_time_delta = (datetime.utcnow() - index_start_time).total_seconds()
   print('[INFO] secondary index creation done in %f seconds' % index_time_delta)

def extract_secondary_index_counter(cfg):
   start_index = (1<<32) - 1
   end_index = -(1<<32)
   for process_id in range(1, cfg['offset']+1):
      file_name = cfg['stats_file_name']+'-'+str(process_id)
      file_path = os.path.join(cfg['offline_index_storage'], file_name)
      with open(file_path, 'r') as txt_file:
         temp_entry = txt_file.readline().strip()
         data = temp_entry.split()
         start_index = min(start_index, int(data[0]))
         end_index = max(end_index, int(data[1]))

   if start_index == ((1<<32) - 1) and end_index == -(1<<32):
      return 0
   return start_index, end_index

def build_primary_index(offline_index_storage, purge_secondary_index):
   cfg = get_configuration(offline_index_storage)
   low, high = extract_secondary_index_counter(cfg)
   cfg['offline_index_counter'] = high - low + 1
   allowed_fields = ['primary_index_file_name', 'primary_index_offset_name', 'offline_index_counter', 'primary_index_size', 'debug_mode', 'offline_index_storage']
   print("[CONFIG] primary index creation running with configuration :")
   display_config(cfg, allowed_fields)
   start_time = datetime.utcnow()
   #merge the title maps
   total_pages = merge_title_map_files(cfg)
   #merge the secondary indexes
   total_unique_tokens, primary_index_counter, total_primary_index_size = merge_all_index_files(cfg, delete_temp_files=purge_secondary_index)
   #finally write to stats file
   stats_file_path = os.path.join(offline_index_storage, cfg['primary_stats_file_name'])
   print
   with open(stats_file_path, 'w', encoding='utf-8') as txt_file:
      txt_file.write("%s %s %s %s\n" % (total_pages, total_unique_tokens, primary_index_counter, total_primary_index_size))
   time_delta = (datetime.utcnow() - start_time).total_seconds()
   print('[INFO] primary index creation done in %.2f seconds' % (time_delta))

def display_stats(offline_index_storage, start_time):
   cfg = get_configuration(offline_index_storage)
   temp_start, temp_end = extract_secondary_index_counter(cfg)
   secondary_index_size = 0.0    #in MB
   primary_index_size = 0.0      #in MB
   title_map_size = 0.0          #in MB
   primary_index_file_count = 0
   secondary_index_file_count = temp_end - temp_start + 1
   total_page_count = 0
   total_unique_word_count = 0
   time_delta = (datetime.utcnow() - start_time).total_seconds()

   #read stats for primary index
   with open(os.path.join(cfg['offline_index_storage'], cfg['primary_stats_file_name'])) as txt_file:
      temp_entry = txt_file.readline()
      data = temp_entry.split()
      total_page_count = int(data[0])
      total_unique_word_count = int(data[1])
      primary_index_file_count = int(data[2])
      primary_index_size = float(data[3])
   
   #read stats for secondary index
   for i in range(1, cfg['offset']+1):
      file_name = cfg['stats_file_name']+'-'+str(i)
      with open(os.path.join(cfg['offline_index_storage'], file_name)) as txt_file:
         temp_entry = txt_file.readline().strip()
         data = temp_entry.split()
         secondary_index_size += int(data[2])/float(1<<20)

   #read stats for title map file
   title_file_size = os.path.getsize(os.path.join(cfg['offline_index_storage'], cfg['doc_title_map']))/float(1<<20)

   print("[INFO] temporary index file count : %s, primary index file count : %s" % (secondary_index_file_count, primary_index_file_count))
   print("[INFO] created inverted index with %s words in %s documents in %.2f seconds" % (total_unique_word_count, total_page_count, time_delta))
   
   if secondary_index_size < (1<<10):
      print("[INFO] total temporary index file size : %.2f MB" % secondary_index_size)
   else:
      print("[INFO] total temporary index file size : %.2f GB" % (secondary_index_size/float(1<<10)))

   if title_file_size < (1<<10):
      print("[INFO] total title map file size : %.2f MB" % title_file_size)
   else:
      print("[INFO] total title map file size : %.2f GB" % (title_file_size/float(1<<10)))

   if primary_index_size < (1<<10):
      print("[INFO] total primary index file size : %.2f MB" % primary_index_size)
   else:
      print("[INFO] total primary index file size : %.2f GB" % (primary_index_size/float(1<<10)))

def initialize(create_secondary_index=True, create_primary_index=True, purge_secondary_index=False):
   #check for the runtime arguments
   if len(sys.argv) != 3:
      print("error : invalid number of argument passed [2] arguments required, need path to xml dump file and path to index_folder")
      return -1
   
   start_time = datetime.utcnow()
   xml_dump_file = os.path.abspath(sys.argv[1])
   offline_index_storage = os.path.abspath(sys.argv[2])

   if create_secondary_index:
      build_secondary_index(xml_dump_file, offline_index_storage, empty_dir=False)
   
   if create_primary_index:
      build_primary_index(offline_index_storage, purge_secondary_index)

   display_stats(offline_index_storage, start_time)

if __name__ == "__main__":
   initialize(create_secondary_index=False, create_primary_index=True)