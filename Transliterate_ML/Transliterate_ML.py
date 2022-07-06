# This Python file uses the following encoding: utf-8
import nltk
# nltk.download('all')
import regex as re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
# from englisttohindi.englisttohindi import EngtoHindi
# nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import model

data=[]
sentences=["My name is Aahaana","The person setting beside me is Aanchal","That person is Cherika","My name is Madhav Kanda","I am working under Prof. Shiva Kumar","I sit beside Narendra Modi sir","Anusha mam is sitting there","Nivedita","Ritika mam is over there","I am working under Ramakrishnan","I work with Krishna"]

file_results=open("Results/results.txt", "a", encoding='utf8')
file_results.flush()

# hindi2engdict={"Aahaana": "आहना","Aaloka":"आलोका","Aamani":"आमनि","Aanchal":"आंचल","Aaravi":"आरवि","Aashi":"आशी","Aayushi":"अयुशि","Adriti":"आद्रिति","Advika":"आद्विका","Aisha":"आयशा","Asin":"आसिन्","Aysha":"आयशा","Bani":"बानी","Bhakti":"भक्ति" ,"Bhaumi":"बाहुमी","Chaitali":"चैताली","Chanakshi":"चानाक्षी","Charul" :"चारूल","Cherika":"चेरिका"}

onechardict={'k': 'K', 'g': 'G', 'j': 'JH', 'z': 'Z', 'T': 'T', 'D': 'D', 'N': 'EN', 't': 'DX', 'd': 'DH', 'n': 'N', 'p': 'P', 'b': 'B',  'm': 'M', 'y': 'Y', 'r': 'R', 'l': 'L', 'v': 'V', 's': 'S', 'h': 'H','a': 'AE','i': 'IH','u': 'UH','e': 'EH','o': 'AO', }

twochardict={'aa': 'AA', 'ee': 'IY', 'oo': 'UW', 'ai': 'EY', 'au': 'OW', 'ri': 'RIH','kh': 'K','gh': 'G','ch': 'CH','jh': 'JH','Th': 'TH','Dh': 'D','th': 'TH','dh': 'D','ph': 'F','bh': 'B','sh': 'SH','Sh': 'SH'}

onecharASCII={'k': '75', 'g': '71', 'j': '7472', 'z': '90', 'T': '84', 'D': '68', 'N': '6978', 't': '6888', 'd': '6872', 'n': '78', 'p': '80', 'b': '66', 'm': '77', 'y': '89', 'r': '82', 'l': '76', 'v': '86', 's': '83', 'h': '72', 'a': '6569', 'i': '7372', 'u': '8572', 'e': '6972', 'o': '6579'}

twocharASCII={'aa': '6565', 'ee': '7389', 'oo': '8587', 'ai': '6989', 'au': '7987', 'ri': '82327372', 'kh': '75', 'gh': '71', 'ch': '6772', 'jh': '7472', 'Th': '8472', 'Dh': '68', 'th': '8472', 'dh': '68', 'ph': '70', 'bh': '66', 'sh': '8372', 'Sh': '8372'}



convert={
    
    'ॐ':'om','अ':'a','आ':'aa','आँ':'aa(nv)','इ':'i','ई':'ee','उ':'u','ऊ':'oo','ऊँ':'oo(nv)','ए':'e','एँ':'e(nv)','ऐ':'ai','ओ':'o','औ':'au','ऋ':'ri','ॠ':'ri','अं':'a(nv)','क्': 'k','क': 'ka','कृ': 'kri', 'का': 'kaa','काँ': 'kaa(nv)', 'कि': 'ki', 'की': 'kee', 'कु': 'ku', 'कू': 'koo', 'के': 'ke', 'कै': 'kai', 'को': 'ko', 'कौ': 'kau','कां': 'kaa(nv)', 'कः': 'kah', 'ख्': 'kh','ख': 'kha','खृ': 'khri', 'खा': 'khaa','खाँ': 'khaa(nv)','खां': 'khaa(nv)', 'खि': 'khi', 'खी': 'khee', 'खु': 'khu', 'खू': 'khoo', 'खे': 'khe', 'खै': 'khai', 'खो': 'kho', 'खौ': 'khau', 'खः': 'khah', 'ग्': 'g','ग': 'ga','गृ': 'gri', 'गा': 'gaa','गाँ': 'gaa(nv)', 'गि': 'gi', 'गी': 'gee', 'गु': 'gu', 'गू': 'goo', 'गे': 'ge', 'गै': 'gai', 'गो': 'go', 'गौ': 'gau','गः': 'gah', 'घ्': 'gh','घ': 'gha','घृ': 'ghri',  'घां': 'ghaa(nv)','घा': 'ghaa','घाँ': 'ghaa(nv)', 'घि': 'ghi', 'घी': 'ghee', 'घु': 'ghu', 'घू': 'ghoo', 'घे': 'ghe', 'घै': 'ghai', 'घो': 'gho', 'घौ': 'ghau','घः': 'ghah','ङ':'nga','ङ्':'ng',
'च्': 'ch','चृ': 'chri','च': 'cha', 'चा': 'chaa','चाँ': 'chaa(nv)','चां': 'chaa(nv)', 'चि': 'chi', 'ची': 'chee', 'चु': 'chu', 'चू': 'choo', 'चे': 'che', 'चै': 'chai', 'चो': 'cho', 'चौ': 'chau', 'चः': 'chah', 'छ्': 'chh','छ': 'chha','छृ': 'chhri', 'छां': 'chhaa(nv)','छा': 'chhaa','छाँ': 'chhaa(nv)', 'छि': 'chhi', 'छी': 'chhee', 'छु': 'chhu', 'छू': 'chhoo', 'छे': 'chhe', 'छै': 'chhai', 'छो': 'chho', 'छौ': 'chhau','छः': 'chhah', 'ज्': 'j','ज': 'ja','जृ': 'jri', 'जा': 'jaa','जाँ': 'jaa(nv)','जां': 'jaa(nv)', 'जि': 'ji', 'जी': 'jee', 'जु': 'ju', 'जू': 'joo', 'जे': 'je', 'जै': 'jai', 'जो': 'jo', 'जौ': 'jau','जः': 'jah','ज़्':'z','ज़':'za','ज़ृ':'zri','ज़ा':'zaa','ज़ाँ':'zaa(nv)','ज़ां':'zaa(nv)','ज़ि':'zi','ज़ी':'zee','ज़ु':'zu','ज़ू':'zoo','ज़े':'ze','ज़ै':'zai','ज़ो':'zo','ज़ौ':'zau','ज़ं':'za(nv)','ज़ः':'zaah','झ्': 'jh','झृ': 'jhri','झ': 'jha', 'झां': 'jhaa(nv)','झा': 'jhaa','झाँ': 'jhaa(nv)', 'झि': 'jhi', 'झी': 'jhee', 'झु': 'jhu', 'झू': 'jhoo', 'झे': 'jhe', 'झै': 'jhai', 'झो': 'jho', 'झौ': 'jhau','झः': 'jhah','ञ':'nja','ञ्':'nj',

'ट्': 'T','ट': 'Ta','टृ': 'Tri', 'टां': 'Taa(nv)','टा': 'Taa','टाँ': 'Taa(nv)','टि': 'Ti', 'टी': 'Tee', 'टु': 'Tu', 'टू': 'Too', 'टे': 'Te', 'टै': 'Tai', 'टो': 'To', 'टौ': 'Tau', 'टः': 'Tah', 'ठ्': 'Th','ठ': 'Tha','ठृ': 'Thri', 'ठाँ': 'Thaa(nv)', 'ठा': 'Thaa','ठां': 'Thaa(nv)', 'ठि': 'Thi', 'ठी': 'Thee', 'ठु': 'Thu', 'ठू': 'Thoo', 'ठे': 'The', 'ठै': 'Thai', 'ठो': 'Tho', 'ठौ': 'Thau', 'ठः': 'Thah', 'ड्': 'D','डृ': 'Dri','ड': 'Da', 'डां': 'Daa(nv)','डा': 'Daa','डाँ': 'Daa(nv)', 'डि': 'Di', 'डी': 'Dee', 'डु': 'Du', 'डू': 'Doo', 'डे': 'De', 'डै': 'Dai', 'डो': 'Do', 'डौ': 'Dau', 'डः': 'Dah', 'ढ्': 'Dh','ढ': 'Dha','ढृ': 'Dhri', 'ढां': 'Dhaa(nv)','ढाँ': 'Dhaa(nv)','ढा': 'Dhaa', 'ढि': 'Dhi', 'ढी': 'Dhee', 'ढु': 'Dhu', 'ढू': 'Dhoo', 'ढे': 'Dhe', 'ढै': 'Dhai', 'ढो': 'Dho', 'ढौ': 'Dhau','ढः': 'Dhah', 'ण्': 'N','णृ': 'Nri','ण': 'Na', 'णां': 'Naa(nv)','णा': 'Naa','णाँ': 'Naa(nv)', 'णि': 'Ni', 'णी': 'Nee', 'णु': 'Nu', 'णू': 'Noo', 'णे': 'Ne', 'णै': 'Nai', 'णो': 'No', 'णौ': 'Nau', 'णः': 'Nah',

'त्': 't','तृ': 'tri','त': 'ta', 'तां': 'taa(nv)','ता': 'taa','ताँ': 'taa(nv)', 'ति': 'ti', 'ती': 'tee', 'तु': 'tu', 'तू': 'tu', 'ते': 'te', 'तै': 'tai', 'तो': 'to', 'तौ': 'tau','तः': 'tah', 'थ्': 'th','थ': 'tha','थृ': 'thri', 'थां': 'thaa(nv)','था': 'thaa','थाँ': 'thaa(nv)', 'थि': 'thi', 'थी': 'thee', 'थु': 'thu', 'थू': 'thoo', 'थे': 'the', 'थै': 'thai', 'थो': 'tho', 'थौ': 'thau','थः': 'thah', 'द्': 'd','दृ': 'dri','द': 'da', 'दां': 'daa(nv)','दा': 'daa','दाँ': 'daa(nv)', 'दि': 'di', 'दी': 'dee', 'दु': 'du', 'दू': 'doo', 'दे': 'de', 'दै': 'dai', 'दो': 'do', 'दौ': 'dau', 'दः': 'dah', 'ध्': 'dh','धृ': 'dhri','ध': 'dha', 'धां': 'dhaa(nv)','धा': 'dhaa','धाँ': 'dhaa(nv)', 'धि': 'dhi', 'धी': 'dhee', 'धु': 'dhu', 'धू': 'dhoo', 'धे': 'dhe', 'धै': 'dhai', 'धो': 'dho', 'धौ': 'dhau',  'धः': 'dhah', 'न्': 'n','न': 'na','नृ': 'nri',  'नां': 'naa(nv)','ना': 'naa','नाँ': 'naa(nv)', 'नि': 'ni', 'नी': 'nee', 'नु': 'nu', 'नू': 'noo', 'ने': 'ne', 'नै': 'nai', 'नो': 'no', 'नौ': 'nau', 'नः': 'nah',

'प्': 'p','पृ': 'pri','प': 'pa','पाँ': 'paa(nv)','पा': 'paa','पां': 'paa(nv)', 'पि': 'pi', 'पी': 'pee', 'पु': 'pu', 'पू': 'poo', 'पे': 'pe', 'पै': 'pai', 'पो': 'po', 'पौ': 'pau', 'पः': 'pah', 'फ्': 'ph','फ': 'pha','फृ': 'phri', 'फां': 'phaa(nv)','फाँ': 'phaa(nv)','फा': 'phaa', 'फि': 'phi', 'फी': 'phee', 'फु': 'phu', 'फू': 'phoo', 'फे': 'phe', 'फै': 'phai', 'फो': 'pho', 'फौ': 'phau', 'फः': 'phah', 'ब्': 'b','बृ': 'bri','ब': 'ba', 'बां': 'baa(nv)','बा': 'baa','बाँ': 'baa(nv)', 'बि': 'bi', 'बी': 'bee', 'बु': 'bu', 'बू': 'boo', 'बे': 'be', 'बै': 'bai', 'बो': 'bo', 'बौ': 'bau','बः': 'bah', 'भ्': 'bh','भ': 'bha', 'भृ': 'bhri', 'भाँ': 'bhaa(nv)','भा': 'bhaa','भां': 'bhaa(nv)', 'भि': 'bhi', 'भी': 'bhee', 'भु': 'bhu', 'भू': 'bhoo', 'भे': 'bhe', 'भै': 'bhai', 'भो': 'bho', 'भौ': 'bhau', 'भः': 'bhah', 'म्': 'm','म': 'ma','मृ': 'mri', 'मां': 'maa(nv)','मा': 'maa','माँ': 'maa(nv)', 'मि': 'mi', 'मी': 'mee', 'मु': 'mu', 'मू': 'moo', 'मे': 'me', 'मै': 'mai', 'मो': 'mo', 'मौ': 'mau', 'मः': 'mah',

'य्': 'y','य': 'ya','यृ': 'yri', 'यां': 'yaa(nv)','या': 'yaa','याँ': 'yaa(nv)', 'यि': 'yi', 'यी': 'yee', 'यु': 'yu', 'यू': 'yoo', 'ये': 'ye', 'यै': 'yai', 'यो': 'yo', 'यौ': 'yau', 'यः': 'yah', 'र्': 'r','र': 'ra','रृ': 'rri', 'रां': 'raa(nv)','रा': 'raa','राँ': 'raa(nv)', 'रि': 'ri', 'री': 'ree', 'रु': 'ru', 'रू': 'roo', 'रे': 're', 'रै': 'rai', 'रो': 'ro', 'रौ': 'rau','रः': 'rah', 'ल्': 'l','ल': 'la','लृ': 'lri', 'लां': 'laa(nv)','ला': 'laa','लाँ': 'laa(nv)', 'लि': 'li', 'ली': 'lee', 'लु': 'lu', 'लू': 'loo', 'ले': 'le', 'लै': 'lai', 'लो': 'lo', 'लौ': 'lau', 'लः': 'lah', 'व्': 'v','व': 'va','वृ': 'vri', 'वां': 'vaa(nv)','वा': 'vaa','वाँ': 'vaa(nv)', 'वि': 'vi', 'वी': 'vee', 'वु': 'vu', 'वू': 'voo', 'वे': 've', 'वै': 'vai', 'वो': 'vo', 'वौ': 'vau', 'वः': 'vah',

'श्': 'sh','शृ': 'shri','श': 'sha', 'शां': 'shaa(nv)','शा': 'shaa','शाँ': 'shaa(nv)', 'शि': 'shi', 'शी': 'shee', 'शु': 'shu', 'शू': 'shoo', 'शे': 'she', 'शै': 'shai', 'शो': 'sho', 'शौ': 'shau', 'शः': 'shah', 'ष्': 'Sh','षृ': 'Shri', 'ष': 'Sha', 'षां': 'Shaa(nv)','षा': 'Shaa','षाँ': 'Shaa(nv)', 'षि': 'Shi', 'षी': 'Shee', 'षु': 'Shu', 'षू': 'Shoo', 'षे': 'She', 'षै': 'Shai', 'षो': 'Sho', 'षौ': 'Shau', 'षः': 'Shah', 'स्': 's','स': 'sa','सृ': 'sri', 'सां': 'saa(nv)','सा': 'saa','साँ': 'saa(nv)', 'सि': 'si', 'सी': 'see', 'सु': 'su', 'सू': 'soo', 'से': 'se', 'सै': 'sai', 'सो': 'so', 'सौ': 'sau', 'सः': 'sah', 'ह्': 'h','हृ': 'hri','ह': 'ha', 'हाँ': 'haa(nv)','हा': 'haa','हां': 'haa(nv)', 'हि': 'hi', 'ही': 'hee', 'हु': 'hu', 'हू': 'hoo', 'हे': 'he', 'है': 'hai', 'हो': 'ho', 'हौ': 'hau','हः': 'hah',

'क्ष्': 'ksh','क्ष': 'ksha','क्षृ': 'kshri', 'क्षा': 'kshaa','क्षाँ': 'kshaa(nv)','क्षां': 'kshaa(nv)', 'क्षि': 'kshi', 'क्षी': 'kshee', 'क्षु': 'kshu', 'क्षू': 'kshoo', 'क्षे': 'kshe', 'क्षै': 'kshai', 'क्षो': 'ksho', 'क्षौ': 'kshau', 'क्षं': 'ksha(nv)', 'क्षः': 'kshah', 'त्र्': 'tr','त्र': 'tra','त्रृ': 'trri',  'त्रा': 'traa','त्राँ': 'traa(nv)','त्रां': 'traa(nv)', 'त्रि': 'tri', 'त्री': 'tree', 'त्रु': 'tru', 'त्रू': 'troo', 'त्रे': 'tre', 'त्रै': 'trai', 'त्रो': 'tro', 'त्रौ': 'trau', 'त्रं': 'tra(nv)', 'त्रः': 'trah', 'ज्ञ्': 'gy','ज्ञ': 'gya','ज्ञृ': 'gyri', 'ज्ञां': 'gyaa(nv)','ज्ञाँ': 'gyaa(nv)','ज्ञा': 'gyaa', 'ज्ञि': 'gyi', 'ज्ञी': 'gyee', 'ज्ञु': 'gyu', 'ज्ञू': 'gyoo', 'ज्ञे': 'gye', 'ज्ञै': 'gyai', 'ज्ञो': 'gyo', 'ज्ञौ': 'gyau', 'ज्ञं': 'gya(nv)', 'ज्ञः': 'gyah',

'कॉ': 'ko', 'खॉ': 'kho', 'गॉ': 'go', 'घॉ': 'gho', 'चॉ': 'cho', 'छॉ': 'chho', 'जॉ': 'jo', 'झॉ': 'jho', 'टॉ': 'To', 'ठॉ': 'Tho', 'डॉ': 'Do', 'ढॉ': 'Dho', 'णॉ': 'No', 'तॉ': 'to', 'थॉ': 'tho', 'दॉ': 'do', 'धॉ': 'dho', 'नॉ': 'no', 'पॉ': 'po', 'फॉ': 'pho', 'बॉ': 'bo', 'भॉ': 'bho', 'मॉ': 'mo', 'यॉ': 'yo', 'रॉ': 'ro', 'लॉ': 'lo', 'षॉ': 'Sho', 'सॉ': 'so', 'वॉ': 'vo', 'शॉ': 'sho', 'हॉ': 'ho',

"/":"(nv)",

'कृं': 'kri(nv)', 'कां': 'kaa(nv)', 'किं': 'ki(nv)', 'कीं': 'kee(nv)', 'कुं': 'ku(nv)', 'कूं': 'koo(nv)', 'कें': 'ke(nv)', 'कैं': 'kai(nv)', 'कों': 'ko(nv)', 'कौं': 'kau(nv)','खृं': 'khri(nv)', 'खां': 'khaa(nv)', 'खिं': 'khi(nv)', 'खीं': 'khee(nv)', 'खुं': 'khu(nv)', 'खूं': 'khoo(nv)', 'खें': 'khe(nv)', 'खैं': 'khai(nv)', 'खों': 'kho(nv)', 'खौं': 'khau(nv)', 'गृं': 'gri(nv)', 'गां': 'gaa(nv)', 'गिं': 'gi(nv)', 'गीं': 'gee(nv)', 'गुं': 'gu(nv)', 'गूं': 'goo(nv)', 'गें': 'ge(nv)', 'गैं': 'gai(nv)', 'गों': 'go(nv)', 'गौं': 'gau(nv)', 'घृं': 'ghri(nv)', 'घां': 'ghaa(nv)', 'घिं': 'ghi(nv)', 'घीं': 'ghee(nv)', 'घुं': 'ghu(nv)', 'घूं': 'ghoo(nv)', 'घें': 'ghe(nv)', 'घैं': 'ghai(nv)', 'घों': 'gho(nv)', 'घौं': 'ghau(nv)',

'चृं': 'chri(nv)','चां': 'chaa(nv)', 'चिं': 'chi(nv)', 'चीं': 'chee(nv)', 'चुं': 'chu(nv)', 'चूं': 'choo(nv)', 'चें': 'che(nv)', 'चैं': 'chai(nv)', 'चों': 'cho(nv)', 'चौं': 'chau(nv)', 'चःं': 'chah(nv)', 'छ्ं': 'chh(nv)',  'छृं': 'chhri(nv)', 'छां': 'chhaa(nv)', 'छिं': 'chhi(nv)', 'छीं': 'chhee(nv)', 'छुं': 'chhu(nv)', 'छूं': 'chhoo(nv)', 'छें': 'chhe(nv)', 'छैं': 'chhai(nv)', 'छों': 'chho(nv)', 'छौं': 'chhau(nv)', 'छःं': 'chhah(nv)', 'ज्ं': 'j(nv)', 'जृं': 'jri(nv)', 'जां': 'jaa(nv)', 'जिं': 'ji(nv)', 'जीं': 'jee(nv)', 'जुं': 'ju(nv)', 'जूं': 'joo(nv)', 'जें': 'je(nv)', 'जैं': 'jai(nv)', 'जों': 'jo(nv)', 'जौं': 'jau(nv)', 'जःं': 'jah(nv)', 'ज़्ं': 'z(nv)', 'ज़ं': 'za(nv)', 'ज़ृं': 'zri(nv)', 'ज़ां': 'zaa(nv)', 'ज़िं': 'zi(nv)', 'ज़ीं': 'zee(nv)', 'ज़ुं': 'zu(nv)', 'ज़ूं': 'zoo(nv)', 'ज़ें': 'ze(nv)', 'ज़ैं': 'zai(nv)', 'ज़ों': 'zo(nv)', 'ज़ौं': 'zau(nv)', 'ज़ःं': 'zaah(nv)', 'झ्ं': 'jh(nv)', 'झृं': 'jhri(nv)', 'झां': 'jhaa(nv)', 'झिं': 'jhi(nv)', 'झीं': 'jhee(nv)', 'झुं': 'jhu(nv)', 'झूं': 'jhoo(nv)', 'झें': 'jhe(nv)', 'झैं': 'jhai(nv)', 'झों': 'jho(nv)', 'झौं': 'jhau(nv)', 'झःं': 'jhah(nv)', 'ट्ं': 'T(nv)', 'टृं': 'Tri(nv)', 'टां': 'Taa(nv)', 'टिं': 'Ti(nv)', 'टीं': 'Tee(nv)', 'टुं': 'Tu(nv)', 'टूं': 'Too(nv)', 'टें': 'Te(nv)', 'टैं': 'Tai(nv)', 'टों': 'To(nv)', 'टौं': 'Tau(nv)', 'टःं': 'Tah(nv)', 'ठ्ं': 'Th(nv)',  'ठृं': 'Thri(nv)', 'ठां': 'Thaa(nv)', 'ठिं': 'Thi(nv)', 'ठीं': 'Thee(nv)', 'ठुं': 'Thu(nv)', 'ठूं': 'Thoo(nv)', 'ठें': 'The(nv)', 'ठैं': 'Thai(nv)', 'ठों': 'Tho(nv)', 'ठौं': 'Thau(nv)', 'ठःं': 'Thah(nv)', 'ड्ं': 'D(nv)', 'डृं': 'Dri(nv)',  'डां': 'Daa(nv)', 'डिं': 'Di(nv)', 'डीं': 'Dee(nv)', 'डुं': 'Du(nv)', 'डूं': 'Doo(nv)', 'डें': 'De(nv)', 'डैं': 'Dai(nv)', 'डों': 'Do(nv)', 'डौं': 'Dau(nv)', 'डःं': 'Dah(nv)', 'ढ्ं': 'Dh(nv)', 'ढृं': 'Dhri(nv)', 'ढां': 'Dhaa(nv)', 'ढिं': 'Dhi(nv)', 'ढीं': 'Dhee(nv)', 'ढुं': 'Dhu(nv)', 'ढूं': 'Dhoo(nv)', 'ढें': 'Dhe(nv)', 'ढैं': 'Dhai(nv)', 'ढों': 'Dho(nv)', 'ढौं': 'Dhau(nv)', 'ढःं': 'Dhah(nv)', 'ण्ं': 'N(nv)', 'णृं': 'Nri(nv)', 'णां': 'Naa(nv)', 'णिं': 'Ni(nv)', 'णीं': 'Nee(nv)', 'णुं': 'Nu(nv)', 'णूं': 'Noo(nv)', 'णें': 'Ne(nv)', 'णैं': 'Nai(nv)', 'णों': 'No(nv)', 'णौं': 'Nau(nv)', 'णःं': 'Nah(nv)', 'त्ं': 't(nv)', 'तृं': 'tri(nv)',  'तां': 'taa(nv)', 'तिं': 'ti(nv)', 'तीं': 'tee(nv)', 'तुं': 'tu(nv)', 'तूं': 'tu(nv)', 'तें': 'te(nv)', 'तैं': 'tai(nv)', 'तों': 'to(nv)', 'तौं': 'tau(nv)', 'तःं': 'tah(nv)', 'थ्ं': 'th(nv)', 'थृं': 'thri(nv)', 'थां': 'thaa(nv)', 'थिं': 'thi(nv)', 'थीं': 'thee(nv)', 'थुं': 'thu(nv)', 'थूं': 'thoo(nv)', 'थें': 'the(nv)', 'थैं': 'thai(nv)', 'थों': 'tho(nv)', 'थौं': 'thau(nv)', 'थःं': 'thah(nv)', 'द्ं': 'd(nv)', 'दृं': 'dri(nv)', 'दां': 'daa(nv)', 'दिं': 'di(nv)', 'दीं': 'dee(nv)', 'दुं': 'du(nv)', 'दूं': 'doo(nv)', 'दें': 'de(nv)', 'दैं': 'dai(nv)', 'दों': 'do(nv)', 'दौं': 'dau(nv)', 'दःं': 'dah(nv)', 'ध्ं': 'dh(nv)', 'धृं': 'dhri(nv)', 'धां': 'dhaa(nv)', 'धिं': 'dhi(nv)', 'धीं': 'dhee(nv)', 'धुं': 'dhu(nv)', 'धूं': 'dhoo(nv)', 'धें': 'dhe(nv)', 'धैं': 'dhai(nv)', 'धों': 'dho(nv)', 'धौं': 'dhau(nv)', 'धःं': 'dhah(nv)', 'न्ं': 'n(nv)',  'नृं': 'nri(nv)', 'नां': 'naa(nv)', 'निं': 'ni(nv)', 'नीं': 'nee(nv)', 'नुं': 'nu(nv)', 'नूं': 'noo(nv)', 'नें': 'ne(nv)', 'नैं': 'nai(nv)', 'नों': 'no(nv)', 'नौं': 'nau(nv)', 'नःं': 'nah(nv)', 'प्ं': 'p(nv)', 'पृं': 'pri(nv)',  'पां': 'paa(nv)', 'पिं': 'pi(nv)', 'पीं': 'pee(nv)', 'पुं': 'pu(nv)', 'पूं': 'poo(nv)', 'पें': 'pe(nv)', 'पैं': 'pai(nv)', 'पों': 'po(nv)', 'पौं': 'pau(nv)', 'पःं': 'pah(nv)', 'फ्ं': 'ph(nv)', 'फृं': 'phri(nv)', 'फां': 'phaa(nv)', 'फिं': 'phi(nv)', 'फीं': 'phee(nv)', 'फुं': 'phu(nv)', 'फूं': 'phoo(nv)', 'फें': 'phe(nv)', 'फैं': 'phai(nv)', 'फों': 'pho(nv)', 'फौं': 'phau(nv)', 'फःं': 'phah(nv)', 'ब्ं': 'b(nv)', 'बृं': 'bri(nv)', 'बां': 'baa(nv)', 'बिं': 'bi(nv)', 'बीं': 'bee(nv)', 'बुं': 'bu(nv)', 'बूं': 'boo(nv)', 'बें': 'be(nv)', 'बैं': 'bai(nv)', 'बों': 'bo(nv)', 'बौं': 'bau(nv)', 'बःं': 'bah(nv)', 'भ्ं': 'bh(nv)', 'भृं': 'bhri(nv)', 'भां': 'bhaa(nv)', 'भिं': 'bhi(nv)', 'भीं': 'bhee(nv)', 'भुं': 'bhu(nv)', 'भूं': 'bhoo(nv)', 'भें': 'bhe(nv)', 'भैं': 'bhai(nv)', 'भों': 'bho(nv)', 'भौं': 'bhau(nv)', 'भःं': 'bhah(nv)', 'म्ं': 'm(nv)', 'मृं': 'mri(nv)', 'मां': 'maa(nv)', 'मिं': 'mi(nv)', 'मीं': 'mee(nv)', 'मुं': 'mu(nv)', 'मूं': 'moo(nv)', 'में': 'me(nv)', 'मैं': 'mai(nv)', 'मों': 'mo(nv)', 'मौं': 'mau(nv)', 'मःं': 'mah(nv)', 'य्ं': 'y(nv)','यृं': 'yri(nv)', 'यां': 'yaa(nv)', 'यिं': 'yi(nv)', 'यीं': 'yee(nv)', 'युं': 'yu(nv)', 'यूं': 'yoo(nv)', 'यें': 'ye(nv)', 'यैं': 'yai(nv)', 'यों': 'yo(nv)', 'यौं': 'yau(nv)', 'यःं': 'yah(nv)', 'र्ं': 'r(nv)', 'रृं': 'rri(nv)', 'रां': 'raa(nv)', 'रिं': 'ri(nv)', 'रीं': 'ree(nv)', 'रुं': 'ru(nv)', 'रूं': 'roo(nv)', 'रें': 're(nv)', 'रैं': 'rai(nv)', 'रों': 'ro(nv)', 'रौं': 'rau(nv)', 'रःं': 'rah(nv)', 'ल्ं': 'l(nv)',  'लृं': 'lri(nv)', 'लां': 'laa(nv)', 'लिं': 'li(nv)', 'लीं': 'lee(nv)', 'लुं': 'lu(nv)', 'लूं': 'loo(nv)', 'लें': 'le(nv)', 'लैं': 'lai(nv)', 'लों': 'lo(nv)', 'लौं': 'lau(nv)', 'लःं': 'lah(nv)', 'व्ं': 'v(nv)', 'वृं': 'vri(nv)', 'वां': 'vaa(nv)', 'विं': 'vi(nv)', 'वीं': 'vee(nv)', 'वुं': 'vu(nv)', 'वूं': 'voo(nv)', 'वें': 've(nv)', 'वैं': 'vai(nv)', 'वों': 'vo(nv)', 'वौं': 'vau(nv)', 'वःं': 'vah(nv)', 'श्ं': 'sh(nv)', 'शृं': 'shri(nv)', 'शां': 'shaa(nv)', 'शिं': 'shi(nv)', 'शीं': 'shee(nv)', 'शुं': 'shu(nv)', 'शूं': 'shoo(nv)', 'शें': 'she(nv)', 'शैं': 'shai(nv)', 'शों': 'sho(nv)', 'शौं': 'shau(nv)', 'शःं': 'shah(nv)', 'ष्ं': 'Sh(nv)', 'षृं': 'Shri(nv)', 'षां': 'Shaa(nv)', 'षिं': 'Shi(nv)', 'षीं': 'Shee(nv)', 'षुं': 'Shu(nv)', 'षूं': 'Shoo(nv)', 'षें': 'She(nv)', 'षैं': 'Shai(nv)', 'षों': 'Sho(nv)', 'षौं': 'Shau(nv)', 'षःं': 'Shah(nv)', 'स्ं': 's(nv)', 'सृं': 'sri(nv)', 'सां': 'saa(nv)', 'सिं': 'si(nv)', 'सीं': 'see(nv)', 'सुं': 'su(nv)', 'सूं': 'soo(nv)', 'सें': 'se(nv)', 'सैं': 'sai(nv)', 'सों': 'so(nv)', 'सौं': 'sau(nv)', 'सःं': 'sah(nv)', 'ह्ं': 'h(nv)', 'हृं': 'hri(nv)','हां': 'haa(nv)', 'हिं': 'hi(nv)', 'हीं': 'hee(nv)', 'हुं': 'hu(nv)', 'हूं': 'hoo(nv)', 'हें': 'he(nv)', 'हैं': 'hai(nv)', 'हों': 'ho(nv)', 'हौं': 'hau(nv)', 'हःं': 'hah(nv)', 'क्ष्ं': 'ksh(nv)', 'क्षं': 'ksha(nv)', 'क्षृं': 'kshri(nv)', 'क्षां': 'kshaa(nv)', 'क्षिं': 'kshi(nv)', 'क्षीं': 'kshee(nv)', 'क्षुं': 'kshu(nv)', 'क्षूं': 'kshoo(nv)', 'क्षें': 'kshe(nv)', 'क्षैं': 'kshai(nv)', 'क्षों': 'ksho(nv)', 'क्षौं': 'kshau(nv)', 'क्षःं': 'kshah(nv)', 'त्र्ं': 'tr(nv)', 'त्रं': 'tra(nv)', 'त्रृं': 'trri(nv)', 'त्रां': 'traa(nv)', 'त्रिं': 'tri(nv)', 'त्रीं': 'tree(nv)', 'त्रुं': 'tru(nv)', 'त्रूं': 'troo(nv)', 'त्रें': 'tre(nv)', 'त्रैं': 'trai(nv)', 'त्रों': 'tro(nv)', 'त्रौं': 'trau(nv)', 'त्रःं': 'trah(nv)', 'ज्ञ्ं': 'gy(nv)', 'ज्ञं': 'gya(nv)', 'ज्ञृं': 'gyri(nv)', 'ज्ञां': 'gyaa(nv)', 'ज्ञिं': 'gyi(nv)', 'ज्ञीं': 'gyee(nv)', 'ज्ञुं': 'gyu(nv)', 'ज्ञूं': 'gyoo(nv)', 'ज्ञें': 'gye(nv)', 'ज्ञैं': 'gyai(nv)', 'ज्ञों': 'gyo(nv)', 'ज्ञौं': 'gyau(nv)', 'ज्ञःं': 'gyah(nv)','कुँ': 'ku(nv)', 'कूँ': 'koo(nv)', 'खुँ': 'khu(nv)', 'खूँ': 'khoo(nv)', 'गुँ': 'gu(nv)', 'गूँ': 'goo(nv)', 'घुँ': 'ghu(nv)', 'घूँ': 'ghoo(nv)', 'चुँ': 'chu(nv)', 'चूँ': 'choo(nv)', 'छुँ': 'chhu(nv)', 'छूँ': 'chhoo(nv)', 'जुँ': 'ju(nv)', 'जूँ': 'joo(nv)', 'झुँ': 'jhu(nv)', 'झूँ': 'jhoo(nv)', 'टुँ': 'Tu(nv)', 'टूँ': 'Too(nv)', 'ठुँ': 'Thu(nv)', 'ठूँ': 'Thoo(nv)', 'डुँ': 'Du(nv)', 'डूँ': 'Doo(nv)', 'ढुँ': 'Dhu(nv)', 'ढूँ': 'Dhoo(nv)', 'णुँ': 'Nu(nv)', 'णूँ': 'Noo(nv)', 'तुँ': 'tu(nv)', 'तूँ': 'too(nv)', 'थुँ': 'thu(nv)', 'थूँ': 'thoo(nv)', 'दुँ': 'du(nv)', 'दूँ': 'doo(nv)', 'धुँ': 'dhu(nv)', 'धूँ': 'dhoo(nv)', 'नुँ': 'nu(nv)', 'नूँ': 'noo(nv)', 'पुँ': 'pu(nv)', 'पूँ': 'poo(nv)', 'फुँ': 'phu(nv)', 'फूँ': 'phoo(nv)', 'बुँ': 'bu(nv)', 'बूँ': 'boo(nv)', 'भुँ': 'bhu(nv)', 'भूँ': 'bhoo(nv)', 'मुँ': 'mu(nv)', 'मूँ': 'moo(nv)', 'युँ': 'yu(nv)', 'यूँ': 'yoo(nv)', 'रुँ': 'ru(nv)', 'रूँ': 'roo(nv)', 'लुँ': 'lu(nv)', 'लूँ': 'loo(nv)', 'षुँ': 'Shu(nv)', 'षूँ': 'Shoo(nv)', 'सुँ': 'su(nv)', 'सूँ': 'soo(nv)', 'वुँ': 'vu(nv)', 'वूँ': 'voo(nv)', 'शुँ': 'shu(nv)', 'शूँ': 'shoo(nv)', 'हुँ': 'hu(nv)', 'हूँ': 'hoo(nv)','कृः': 'kriha', 'काः': 'kaahaa', 'किः': 'kihi', 'कीः': 'keehee', 'कुः': 'kuhu', 'कूः': 'koohoo', 'केः': 'kehe', 'कैः': 'kaihai', 'कोः': 'koho', 'कौः': 'kauhau', 'खृः': 'khriha', 'खाः': 'khaahaa', 'खिः': 'khihi', 'खीः': 'kheehee', 'खुः': 'khuhu', 'खूः': 'khoohoo', 'खेः': 'khehe', 'खैः': 'khaihai', 'खोः': 'khoho', 'खौः': 'khauhau', 'गृः': 'griha', 'गाः': 'gaahaa', 'गिः': 'gihi', 'गीः': 'geehee', 'गुः': 'guhu', 'गूः': 'goohoo', 'गेः': 'gehe', 'गैः': 'gaihai', 'गोः': 'goho', 'गौः': 'gauhau', 'घृः': 'ghriha', 'घाः': 'ghaahaa', 'घिः': 'ghihi', 'घीः': 'gheehee', 'घुः': 'ghuhu', 'घूः': 'ghoohoo', 'घेः': 'ghehe', 'घैः': 'ghaihai', 'घोः': 'ghoho', 'घौः': 'ghauhau', 'चृः': 'chriha', 'चाः': 'chaahaa', 'चिः': 'chihi', 'चीः': 'cheehee', 'चुः': 'chuhu', 'चूः': 'choohoo', 'चेः': 'chehe', 'चैः': 'chaihai', 'चोः': 'choho', 'चौः': 'chauhau', 'छृः': 'chhriha', 'छाः': 'chhaahaa', 'छिः': 'chhihi', 'छीः': 'chheehee', 'छुः': 'chhuhu', 'छूः': 'chhoohoo', 'छेः': 'chhehe', 'छैः': 'chhaihai', 'छोः': 'chhoho', 'छौः': 'chhauhau', 'जृः': 'jriha', 'जाः': 'jaahaa', 'जिः': 'jihi', 'जीः': 'jeehee', 'जुः': 'juhu', 'जूः': 'joohoo', 'जेः': 'jehe', 'जैः': 'jaihai', 'जोः': 'joho', 'जौः': 'jauhau', 'ज़ृः': 'zriha', 'ज़ाः': 'zaahaa', 'ज़िः': 'zihi', 'ज़ीः': 'zeehee', 'ज़ुः': 'zuhu', 'ज़ूः': 'zoohoo', 'ज़ेः': 'zehe', 'ज़ोः': 'zoho', 'ज़ौः': 'zauhau', 'झृः': 'jhriha', 'झाः': 'jhaahaa', 'झिः': 'jhihi', 'झीः': 'jheehee', 'झुः': 'jhuhu', 'झूः': 'jhoohoo', 'झेः': 'jhehe', 'झैः': 'jhaihai', 'झोः': 'jhoho', 'झौः': 'jhauhau', 'टृः': 'Triha', 'टाः': 'Taahaa', 'टिः': 'Tihi', 'टीः': 'Teehee', 'टुः': 'Tuhu', 'टूः': 'Toohoo', 'टेः': 'Tehe', 'टैः': 'Taihai', 'टोः': 'Toho', 'टौः': 'Tauhau', 'ठृः': 'Thriha', 'ठाः': 'Thaahaa', 'ठिः': 'Thihi', 'ठीः': 'Theehee', 'ठुः': 'Thuhu', 'ठूः': 'Thoohoo', 'ठेः': 'Thehe', 'ठैः': 'Thaihai', 'ठोः': 'Thoho', 'ठौः': 'Thauhau', 'डृः': 'Driha', 'डाः': 'Daahaa', 'डिः': 'Dihi', 'डीः': 'Deehee', 'डुः': 'Duhu', 'डूः': 'Doohoo', 'डेः': 'Dehe', 'डैः': 'Daihai', 'डोः': 'Doho', 'डौः': 'Dauhau', 'ढृः': 'Dhriha', 'ढाः': 'Dhaahaa', 'ढिः': 'Dhihi', 'ढीः': 'Dheehee', 'ढुः': 'Dhuhu', 'ढूः': 'Dhoohoo', 'ढेः': 'Dhehe', 'ढैः': 'Dhaihai', 'ढोः': 'Dhoho', 'ढौः': 'Dhauhau', 'णृः': 'Nriha', 'णाः': 'Naahaa', 'णिः': 'Nihi', 'णीः': 'Neehee', 'णुः': 'Nuhu', 'णूः': 'Noohoo', 'णेः': 'Nehe', 'णैः': 'Naihai', 'णोः': 'Noho', 'णौः': 'Nauhau', 'तृः': 'triha', 'ताः': 'taahaa', 'तिः': 'tihi', 'तीः': 'teehee', 'तुः': 'tuhu', 'तूः': 'tuhoo', 'तेः': 'tehe', 'तैः': 'taihai', 'तोः': 'toho', 'तौः': 'tauhau', 'थृः': 'thriha', 'थाः': 'thaahaa', 'थिः': 'thihi', 'थीः': 'theehee', 'थुः': 'thuhu', 'थूः': 'thoohoo', 'थेः': 'thehe', 'थैः': 'thaihai', 'थोः': 'thoho', 'थौः': 'thauhau', 'दृः': 'driha', 'दाः': 'daahaa', 'दिः': 'dihi', 'दीः': 'deehee', 'दुः': 'duhu', 'दूः': 'doohoo', 'देः': 'dehe', 'दैः': 'daihai', 'दोः': 'doho', 'दौः': 'dauhau', 'धृः': 'dhriha', 'धाः': 'dhaahaa', 'धिः': 'dhihi', 'धीः': 'dheehee', 'धुः': 'dhuhu', 'धूः': 'dhoohoo', 'धेः': 'dhehe', 'धैः': 'dhaihai', 'धोः': 'dhoho', 'धौः': 'dhauhau', 'नृः': 'nriha', 'नाः': 'naahaa', 'निः': 'nihi', 'नीः': 'neehee', 'नुः': 'nuhu', 'नूः': 'noohoo', 'नेः': 'nehe', 'नैः': 'naihai', 'नोः': 'noho', 'नौः': 'nauhau', 'पृः': 'priha', 'पाः': 'paahaa', 'पिः': 'pihi', 'पीः': 'peehee', 'पुः': 'puhu', 'पूः': 'poohoo', 'पेः': 'pehe', 'पैः': 'paihai', 'पोः': 'poho', 'पौः': 'pauhau', 'फृः': 'phriha', 'फाः': 'phaahaa', 'फिः': 'phihi', 'फीः': 'pheehee', 'फुः': 'phuhu', 'फूः': 'phoohoo', 'फेः': 'phehe', 'फैः': 'phaihai', 'फोः': 'phoho', 'फौः': 'phauhau', 'बृः': 'briha', 'बाः': 'baahaa', 'बिः': 'bihi', 'बीः': 'beehee', 'बुः': 'buhu', 'बूः': 'boohoo', 'बेः': 'behe', 'बैः': 'baihai', 'बोः': 'boho', 'बौः': 'bauhau', 'भृः': 'bhriha', 'भाः': 'bhaahaa', 'भिः': 'bhihi', 'भीः': 'bheehee', 'भुः': 'bhuhu', 'भूः': 'bhoohoo', 'भेः': 'bhehe', 'भैः': 'bhaihai', 'भोः': 'bhoho', 'भौः': 'bhauhau', 'मृः': 'mriha', 'माः': 'maahaa', 'मिः': 'mihi', 'मीः': 'meehee', 'मुः': 'muhu', 'मूः': 'moohoo', 'मेः': 'mehe', 'मैः': 'maihai', 'मोः': 'moho', 'मौः': 'mauhau', 'यृः': 'yriha', 'याः': 'yaahaa', 'यिः': 'yihi', 'यीः': 'yeehee', 'युः': 'yuhu', 'यूः': 'yoohoo', 'येः': 'yehe', 'यैः': 'yaihai', 'योः': 'yoho', 'यौः': 'yauhau', 'रृः': 'rriha', 'राः': 'raahaa', 'रिः': 'rihi', 'रीः': 'reehee', 'रुः': 'ruhu', 'रूः': 'roohoo', 'रेः': 'rehe', 'रैः': 'raihai', 'रोः': 'roho', 'रौः': 'rauhau', 'लृः': 'lriha', 'लाः': 'laahaa', 'लिः': 'lihi', 'लीः': 'leehee', 'लुः': 'luhu', 'लूः': 'loohoo', 'लेः': 'lehe', 'लैः': 'laihai', 'लोः': 'loho', 'लौः': 'lauhau', 'वृः': 'vriha', 'वाः': 'vaahaa', 'विः': 'vihi', 'वीः': 'veehee', 'वुः': 'vuhu', 'वूः': 'voohoo', 'वेः': 'vehe', 'वैः': 'vaihai', 'वोः': 'voho', 'वौः': 'vauhau', 'शृः': 'shriha', 'शाः': 'shaahaa', 'शिः': 'shihi', 'शीः': 'sheehee', 'शुः': 'shuhu', 'शूः': 'shoohoo', 'शेः': 'shehe', 'शैः': 'shaihai', 'शोः': 'shoho', 'शौः': 'shauhau', 'षृः': 'Shriha', 'षाः': 'Shaahaa', 'षिः': 'Shihi', 'षीः': 'Sheehee', 'षुः': 'Shuhu', 'षूः': 'Shoohoo', 'षेः': 'Shehe', 'षैः': 'Shaihai', 'षोः': 'Shoho', 'षौः': 'Shauhau', 'सृः': 'sriha', 'साः': 'saahaa', 'सिः': 'sihi', 'सीः': 'seehee', 'सुः': 'suhu', 'सूः': 'soohoo', 'सेः': 'sehe', 'सैः': 'saihai', 'सोः': 'soho', 'सौः': 'sauhau', 'हृः': 'hriha', 'हाः': 'haahaa', 'हिः': 'hihi', 'हीः': 'heehee', 'हुः': 'huhu', 'हूः': 'hoohoo', 'हेः': 'hehe', 'हैः': 'haihai', 'होः': 'hoho', 'हौः': 'hauhau'}

extralist=["क","ख","ग","घ","च","छ","ज","झ","ट","ठ","ड","ढ","ण","त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","ष","स","व","श","ह"]

panchamCorrespondence={"क":'ङ्',"ख":'ङ्',"ग":'ङ्',"घ":'ङ्',"च":'ञ्',"छ":'ञ्',"ज":'ञ्',"झ":'ञ्',"ट":"ण्","ठ":"ण्","ड":"ण्","ढ":"ण्","ण":"ण्","त":"न्","थ":"न्","द":"न्","ध":"न्","न":"न्","प":"म्","फ":"म्","ब":"म्","भ":"म्","म":"म्","य":"/","र":"/","ल":"/","ष":"/","स":"/","व":"/","श":"/","ह":"/"}

['कं', 'खं', 'गं', 'घं', 'चं', 'छं', 'जं', 'झं', 'टं', 'ठं', 'डं', 'ढं', 'णं', 'तं', 'थं', 'दं', 'धं', 'नं', 'पं', 'फं', 'बं', 'भं', 'मं', 'यं', 'रं', 'लं', 'षं', 'सं', 'वं', 'शं', 'हं']


# from numpy import character

stop_words = set(stopwords.words('english'))
for sentence in sentences:
    tokenized = sent_tokenize(sentence)
    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        wordsList = [w for w in wordsList if not w in stop_words]
        tagged = nltk.pos_tag(wordsList)
        for w,tag in tagged:
            if tag=="NNP" or tag=="NNPS":
                data.append(w)

# output_text = transliterate(input_text, sanscript.ITRANS,sanscript.DEVANAGARI)
conversiondict={}
engphonedict={}
for item in data:
    name=model.transliterate(item)
    if item[-1]=='a':
        if name[-1]=='ा':
            pass
        else:
            name+='ा'
    print(name)
    file_results.write(name)
    file_results.write("\n")
    characterlist=re.findall(r'\X',name)
    conString=''
    n=len(characterlist)
    for i in range(n):
        # print(characterlist[i])
        if(i==n-1):
            try:
                finalchar=characterlist[i]
                flag=0
                for element in extralist:
                    if(finalchar==element):
                        flag=1
                if flag==1:
                    conString+=convert[characterlist[i]]
                    conString=conString[:-1]
                else:
                    conString+=convert[characterlist[i]]
            except KeyError:
                if(characterlist[i][-1]=="ं"):
                    characterhalf=characterlist[i][0]
                    conString+=convert[characterhalf]+convert['म्']
        else:
            try:
                conString+=convert[characterlist[i]]
            except KeyError:
                if(characterlist[i][-1]=="ं"):
                    characterhalf=characterlist[i][0]
                    conString+=convert[characterhalf]+convert[panchamCorrespondence[characterlist[i+1][0]]]
    conversiondict[name]=conString

    start=0
    EngPhostr=''
    while(start!=len(conString)):
        flag=0 
        twochar=conString[start:start+2]
        onechar=conString[start]
        if conString[start:start+3]=='chh':
            EngPhostr+='6772'
            EngPhostr+='32'
            # Code corresponding to the space is 32
            start+=3
            flag=1
        if conString[start:start+4]=='(nv)':
            EngPhostr+='7886'
            # Above ascii code corresponds to the NV character
            EngPhostr+='32'
            # Code corresponding to the space is 32
            start+=4
            flag=1
        if flag==0: 
            for key,value in twocharASCII.items():
                if key==twochar:
                    EngPhostr+=value
                    EngPhostr+='32'
                    # Code corresponding to the space is 32
                    start+=2
                    flag=1
        if flag==0:
            for key,value in onecharASCII.items():
                if key==onechar:
                    EngPhostr+=value
                    EngPhostr+='32'
                    start+=1
                    flag=1
    engphonedict[conString]=EngPhostr
        
file = open("results.bin", "wb")
for key,element in engphonedict.items():
    key=bytes(key, encoding='utf8')
    newkeyByte=bytearray(key)
    file.write(newkeyByte)
    file.write(b":")
    element=bytes(element, encoding='utf8')
    newelementByte=bytearray(element)
    file.write(element)
    file.write(b"\n")
file.close()
file_results.close()
print(engphonedict)
