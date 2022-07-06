# This Python file uses the following encoding: utf-8

extralist=["क","ख","ग","घ","च","छ","ज","झ","ट","ठ","ड","ढ","ण","त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","ष","स","व","श","ह"]
convert={'क': 'ka','कृ': 'kri', 'का': 'kaa','कि': 'ki', 'की': 'kee', 'कु': 'ku', 'कू': 'koo', 'के': 'ke', 'कै': 'kai', 'को': 'ko', 'कौ': 'kau','ख': 'kha','खृ': 'khri', 'खा': 'khaa','खि': 'khi', 'खी': 'khee', 'खु': 'khu', 'खू': 'khoo', 'खे': 'khe', 'खै': 'khai', 'खो': 'kho', 'खौ': 'khau','ग': 'ga','गृ': 'gri', 'गा': 'gaa','गि': 'gi', 'गी': 'gee', 'गु': 'gu', 'गू': 'goo', 'गे': 'ge', 'गै': 'gai', 'गो': 'go', 'गौ': 'gau','घ': 'gha','घृ': 'ghri',  'घा': 'ghaa','घि': 'ghi', 'घी': 'ghee', 'घु': 'ghu', 'घू': 'ghoo', 'घे': 'ghe', 'घै': 'ghai', 'घो': 'gho', 'घौ': 'ghau','चृ': 'chri','च': 'cha', 'चा': 'chaa', 'चि': 'chi', 'ची': 'chee', 'चु': 'chu', 'चू': 'choo', 'चे': 'che', 'चै': 'chai', 'चो': 'cho', 'चौ': 'chau',  'चः': 'chah', 'छ्': 'chh','छ': 'chha','छृ': 'chhri', 'छा': 'chhaa','छि': 'chhi', 'छी': 'chhee', 'छु': 'chhu', 'छू': 'chhoo', 'छे': 'chhe', 'छै': 'chhai', 'छो': 'chho', 'छौ': 'chhau',  'छः': 'chhah', 'ज्': 'j','ज': 'ja','जृ': 'jri', 'जा':'jaa', 'जि': 'ji', 'जी': 'jee', 'जु': 'ju', 'जू': 'joo', 'जे': 'je', 'जै': 'jai', 'जो': 'jo', 'जौ': 'jau',  'जः': 'jah','ज़्':'z','ज़':'za','ज़ृ':'zri','ज़ा':'zaa','ज़ि':'zi','ज़ी':'zee','ज़ु':'zu','ज़ू':'zoo','ज़े':'ze','ज़ै':'zai','ज़ो':'zo','ज़ौ':'zau','ज़ः':'zaah','झ्': 'jh','झृ': 'jhri','झ': 'jha', 'झा': 'jhaa' ,'झि': 'jhi', 'झी': 'jhee', 'झु': 'jhu', 'झू': 'jhoo', 'झे': 'jhe', 'झै': 'jhai', 'झो': 'jho', 'झौ': 'jhau',  'झः': 'jhah', 'ट्': 'T','ट': 'Ta','टृ': 'Tri', 'टा': 'Taa','टि': 'Ti', 'टी': 'Tee', 'टु': 'Tu', 'टू': 'Too', 'टे': 'Te', 'टै': 'Tai', 'टो': 'To', 'टौ': 'Tau',  'टः': 'Tah', 'ठ्': 'Th','ठ': 'Tha','ठृ': 'Thri',  'ठा': 'Thaa', 'ठि': 'Thi', 'ठी': 'Thee', 'ठु': 'Thu', 'ठू': 'Thoo', 'ठे': 'The', 'ठै': 'Thai', 'ठो': 'Tho', 'ठौ': 'Thau',  'ठः': 'Thah', 'ड्': 'D','डृ': 'Dri','ड': 'Da', 'डा': 'Daa', 'डि': 'Di', 'डी': 'Dee', 'डु': 'Du', 'डू': 'Doo', 'डे': 'De', 'डै': 'Dai', 'डो': 'Do', 'डौ': 'Dau',  'डः': 'Dah', 'ढ्': 'Dh','ढ': 'Dha','ढृ': 'Dhri', 'ढा': 'Dhaa', 'ढि': 'Dhi', 'ढी': 'Dhee', 'ढु': 'Dhu', 'ढू': 'Dhoo', 'ढे': 'Dhe', 'ढै': 'Dhai', 'ढो': 'Dho', 'ढौ': 'Dhau',  'ढः': 'Dhah', 'ण्': 'N','णृ': 'Nri','ण': 'Na', 'णा': 'Naa', 'णि': 'Ni', 'णी': 'Nee', 'णु': 'Nu', 'णू': 'Noo', 'णे': 'Ne', 'णै': 'Nai', 'णो': 'No', 'णौ': 'Nau',  'णः': 'Nah', 'त्': 't','तृ': 'tri','त': 'ta', 'ता': 'taa', 'ति': 'ti', 'ती': 'tee', 'तु': 'tu', 'तू': 'tu', 'ते': 'te', 'तै': 'tai', 'तो': 'to', 'तौ': 'tau',  'तः': 'tah', 'थ्': 'th','थ': 'tha','थृ': 'thri', 'था': 'thaa', 'थि': 'thi', 'थी': 'thee', 'थु': 'thu', 'थू': 'thoo', 'थे': 'the', 'थै': 'thai', 'थो': 'tho', 'थौ': 'thau',  'थः': 'thah', 'द्': 'd','दृ': 'dri','द': 'da', 'दा': 'daa', 'दि': 'di', 'दी': 'dee', 'दु': 'du', 'दू': 'doo', 'दे': 'de', 'दै': 'dai', 'दो': 'do', 'दौ': 'dau',  'दः': 'dah', 'ध्': 'dh','धृ': 'dhri','ध': 'dha', 'धा': 'dhaa', 'धि': 'dhi', 'धी': 'dhee', 'धु': 'dhu', 'धू': 'dhoo', 'धे': 'dhe', 'धै': 'dhai', 'धो': 'dho', 'धौ': 'dhau',  'धः': 'dhah', 'न्': 'n','न': 'na','नृ': 'nri',  'ना': 'naa', 'नि': 'ni', 'नी': 'nee', 'नु': 'nu', 'नू': 'noo', 'ने': 'ne', 'नै': 'nai', 'नो': 'no', 'नौ': 'nau',  'नः': 'nah', 'प्': 'p','पृ': 'pri','प': 'pa','पा':'paa','पि': 'pi', 'पी': 'pee', 'पु': 'pu', 'पू': 'poo', 'पे': 'pe', 'पै': 'pai', 'पो': 'po', 'पौ': 'pau',  'पः': 'pah', 'फ्': 'ph','फ': 'pha','फृ': 'phri', 'फा': 'phaa', 'फि': 'phi', 'फी': 'phee', 'फु': 'phu', 'फू': 'phoo', 'फे': 'phe', 'फै': 'phai', 'फो': 'pho', 'फौ': 'phau',  'फः': 'phah', 'ब्': 'b','बृ': 'bri','ब': 'ba', 'बा': 'baa', 'बि': 'bi', 'बी': 'bee', 'बु': 'bu', 'बू': 'boo', 'बे': 'be', 'बै': 'bai', 'बो': 'bo', 'बौ': 'bau',  'बः': 'bah', 'भ्': 'bh','भ': 'bha', 'भृ': 'bhri', 'भा': 'bhaa','भि': 'bhi', 'भी': 'bhee', 'भु': 'bhu', 'भू': 'bhoo', 'भे': 'bhe', 'भै': 'bhai', 'भो': 'bho', 'भौ': 'bhau',  'भः': 'bhah', 'म्': 'm','म': 'ma','मृ': 'mri', 'मा': 'maa', 'मि': 'mi', 'मी': 'mee', 'मु': 'mu', 'मू': 'moo', 'मे': 'me', 'मै': 'mai', 'मो': 'mo', 'मौ': 'mau',  'मः': 'mah', 'य्': 'y','य': 'ya','यृ': 'yri', 'या': 'yaa', 'यि': 'yi', 'यी': 'yee', 'यु': 'yu', 'यू': 'yoo', 'ये': 'ye', 'यै': 'yai', 'यो': 'yo', 'यौ': 'yau',  'यः': 'yah', 'र्': 'r','र': 'ra','रृ': 'rri', 'रा': 'raa', 'रि': 'ri', 'री': 'ree', 'रु': 'ru', 'रू': 'roo', 'रे': 're', 'रै': 'rai', 'रो': 'ro', 'रौ': 'rau',  'रः': 'rah', 'ल्': 'l','ल': 'la','लृ': 'lri', 'ला': 'laa', 'लि': 'li', 'ली': 'lee', 'लु': 'lu', 'लू': 'loo', 'ले': 'le', 'लै': 'lai', 'लो': 'lo', 'लौ': 'lau',  'लः': 'lah', 'व्': 'v','व': 'va','वृ': 'vri', 'वा': 'vaa', 'वि': 'vi', 'वी': 'vee', 'वु': 'vu', 'वू': 'voo', 'वे': 've', 'वै': 'vai', 'वो': 'vo', 'वौ': 'vau',  'वः': 'vah', 'श्': 'sh','शृ': 'shri','श': 'sha', 'शा': 'shaa', 'शि': 'shi', 'शी': 'shee', 'शु': 'shu', 'शू': 'shoo', 'शे': 'she', 'शै': 'shai', 'शो': 'sho', 'शौ': 'shau',  'शः': 'shah', 'ष्': 'Sh','षृ': 'Shri', 'ष': 'Sha', 'षा': 'Shaa', 'षि': 'Shi', 'षी': 'Shee', 'षु': 'Shu', 'षू': 'Shoo', 'षे': 'She', 'षै': 'Shai', 'षो': 'Sho', 'षौ': 'Shau',  'षः': 'Shah', 'स्': 's','स': 'sa','सृ': 'sri', 'सा': 'saa', 'सि': 'si', 'सी': 'see', 'सु': 'su', 'सू': 'soo', 'से': 'se', 'सै': 'sai', 'सो': 'so', 'सौ': 'sau',  'सः': 'sah', 'ह्': 'h','हृ': 'hri','ह': 'ha', 'हा':'haa', 'हि': 'hi', 'ही': 'hee', 'हु': 'hu', 'हू': 'hoo', 'हे': 'he', 'है': 'hai', 'हो': 'ho', 'हौ': 'hau',  'हः': 'hah', 'क्ष्': 'ksh','क्ष': 'ksha','क्षृ': 'kshri', 'क्षा': 'kshaa','क्षि': 'kshi', 'क्षी': 'kshee', 'क्षु': 'kshu', 'क्षू': 'kshoo', 'क्षे': 'kshe', 'क्षै': 'kshai', 'क्षो': 'ksho', 'क्षौ': 'kshau',  'क्षः': 'kshah', 'त्र्': 'tr','त्र': 'tra','त्रृ': 'trri',  'त्रा': 'traa', 'त्रि': 'tri', 'त्री': 'tree', 'त्रु': 'tru', 'त्रू': 'troo', 'त्रे': 'tre', 'त्रै': 'trai', 'त्रो': 'tro', 'त्रौ': 'trau',  'त्रः': 'trah', 'ज्ञ्': 'gy','ज्ञ': 'gya','ज्ञृ': 'gyri', 'ज्ञा': 'gyaa', 'ज्ञि': 'gyi', 'ज्ञी': 'gyee', 'ज्ञु': 'gyu', 'ज्ञू': 'gyoo', 'ज्ञे': 'gye', 'ज्ञै': 'gyai', 'ज्ञो': 'gyo', 'ज्ञौ': 'gyau',  'ज्ञः': 'gyah'}

extradict={'कु': 'ku', 'कू': 'koo', 'खु': 'khu', 'खू': 'khoo', 'गु': 'gu', 'गू': 'goo', 'घु': 'ghu', 'घू': 'ghoo', 'चु': 'chu', 'चू': 'choo', 'छु': 'chhu', 'छू': 'chhoo', 'जु': 'ju', 'जू': 'joo', 'झु': 'jhu', 'झू': 'jhoo', 'टु': 'Tu', 'टू': 'Too', 'ठु': 'Thu', 'ठू': 'Thoo', 'डु': 'Du', 'डू': 'Doo', 'ढु': 'Dhu', 'ढू': 'Dhoo', 'णु': 'Nu', 'णू': 'Noo', 'तु': 'tu', 'तू': 'too', 'थु': 'thu', 'थू': 'thoo', 'दु': 'du', 'दू': 'doo', 'धु': 'dhu', 'धू': 'dhoo', 'नु': 'nu', 'नू': 'noo', 'पु': 'pu', 'पू': 'poo', 'फु': 'phu', 'फू': 'phoo', 'बु': 'bu', 'बू': 'boo', 'भु': 'bhu', 'भू': 'bhoo', 'मु': 'mu', 'मू': 'moo', 'यु': 'yu', 'यू': 'yoo', 'रु': 'ru', 'रू': 'roo', 'लु': 'lu', 'लू': 'loo', 'षु': 'Shu', 'षू': 'Shoo', 'सु': 'su', 'सू': 'soo', 'वु': 'vu', 'वू': 'voo', 'शु': 'shu', 'शू': 'shoo', 'हु': 'hu', 'हू': 'hoo'}


change={'ृ':'ha', 'ा':'haa', 'ि':'hi', 'ी':'hee', 'ु':'hu', 'ू':'hoo', 'े':'he', 'ै':'hai', 'ो':'ho', 'ौ':'hau'}

signs=['', 'ृ', 'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ']


lit=["क","ख","ग","घ","च","छ","ज","झ","ट","ठ","ड","ढ","ण","त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","ष","स","व","श","ह"]
newlist=[]

newlit=['कं', 'खं', 'गं', 'घं', 'चं', 'छं', 'जं', 'झं', 'टं', 'ठं', 'डं', 'ढं', 'णं', 'तं', 'थं', 'दं', 'धं', 'नं', 'पं', 'फं', 'बं', 'भं', 'मं', 'यं', 'रं', 'लं', 'षं', 'सं', 'वं', 'शं', 'हं']



content=['K   k     क्', 'K|HH   kh    ख्', 'G   g     ग्', 'G|HH   gh    घ्', 'CH   ch च्', 'CH|HH  chh   छ्', 'JH    j     ज्', 'Z   z     ज़्', 'JH|HH  jh    झ', 'T   T     ट्', 'TH   Th    ठ्', 'D   D     ड्', 'D|HH   Dh    ढ', 'EN   N     ण', '', 'DX   t     त', 'TH   th    थ्', 'DH   d     द', 'D|HH   dh    ध', 'N   n     न्', '', 'P   p     प्', 'F   ph    फ्', 'B   b ब्', 'B|HH   bh    भ', 'M   m     म् ', 'Y   y     य् ', 'R   r     र्', 'L   l     ल्', 'V   v     व्', 'WH   v     व्', 'SH   sh    श्', 'SH   sh    ष्', 'S   s     स्', 'H[3]   h     ह्']

content2=['AE    a', 'AX    aa', 'IH    i', 'IY    ee', 'UH    u', 'UW    oo', 'EH  e', 'AY      ai', 'AO      o', 'OW      au', 'R|IH    ri']



# datanew='''
# AE	    a       
# AX	    aa
# IH	    i       
# IY	    ee      
# UH	    u       
# UW	    oo      
# EH  	e
# AY      ai
# AO      o       
# OW      au
# R|IH    ri   
# '''.splitlines()

onechardict={'k': 'K', 'g': 'G', 'j': 'JH', 'z': 'Z', 'T': 'T', 'D': 'D', 'N': 'EN', 't': 'DX', 'd': 'DH', 'n': 'N', 'p': 'P', 'b': 'B',  'm': 'M', 'y': 'Y', 'r': 'R', 'l': 'L', 'v': 'V', 's': 'S', 'h': 'H','a': 'AE','i': 'IH','u': 'UH','e': 'EH','o': 'AO', }

onecharASCII={'k': '75', 'g': '71', 'j': '7472', 'z': '90', 'T': '84', 'D': '68', 'N': '6978', 't': '6888', 'd': '6872', 'n': '78', 'p': '80', 'b': '66', 'm': '77', 'y': '89', 'r': '82', 'l': '76', 'v': '86', 's': '83', 'h': '72', 'a': '6569', 'i': '7372', 'u': '8572', 'e': '6972', 'o': '6579'}

twochardict={'aa': 'AA', 'ee': 'IY', 'oo': 'UW', 'ai': 'EY', 'au': 'OW', 'ri': 'R IH','kh': 'K','gh': 'G','ch': 'CH','jh': 'JH','Th': 'TH','Dh': 'D','th': 'TH','dh': 'D','ph': 'F','bh': 'B','sh': 'SH','Sh': 'SH',}

twocharASCII={'aa': '6565', 'ee': '7389', 'oo': '8587', 'ai': '6989', 'au': '7987', 'ri': '82327372', 'kh': '75', 'gh': '71', 'ch': '6772', 'jh': '7472', 'Th': '8472', 'Dh': '68', 'th': '8472', 'dh': '68', 'ph': '70', 'bh': '66', 'sh': '8372', 'Sh': '8372'}

# for key,value in twochardict.items():
#     ascii_value=''
#     for letter in value:
#         ascii_value+=str(ord(letter))
#     twocharASCII[key]=ascii_value

# print(twocharASCII)

# 821247372

# y=content[0]
# x=y.split()
# print(x[0],x[1])

