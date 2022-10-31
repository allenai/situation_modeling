import logging
import re

__all__ = [
    "compute_babi_abstraction",
    "compute_cluttr_abstraction",
]


PEOPLE = {
    "Julie" : "female",
    "Mary"  : "female",
    "Sandra"  : "female",
    "Daniel"  : "male",
    "Bill"    : "male",
    "John"    : "male",
    "Fred"    : "male",
    "Jeff"    : "male",
}

def compute_babi_abstraction(rep1,rep2):
    """Compute an abstract description between two babi semantic representations 

    :param rep1: the first representation
    :param rep2: the second representation 
    :rtype: str
    :returns: a textual description of two babi events
    :raises: ValueError 

    >>> from situation_modeling.utils.dataset_utils import compute_babi_abstraction
    >>> compute_babi_abstraction("move(John,park)", "move(John,kitchen)")
    'John moved somewhere'
    >>> compute_babi_abstraction("move(John,park)", "move(Mary_Jeff,kitchen)")
    'One or more people moved somewhere'
    """
    event1,arg11,arg21 = re.search(r'(.+)\((.+)\,(.+)\)',rep1.replace("_pron","").strip()).groups()
    event2,arg12,arg22 = re.search(r'(.+)\((.+)\,(.+)\)',rep2.replace("_pron","").strip()).groups()
    g1 = [PEOPLE[v] for v in arg11.split("_") if v in PEOPLE]
    g2 = [PEOPLE[v] for v in arg12.split("_") if v in PEOPLE]
    first_arg = [a.strip() for a in arg11.split(',')]
    second_arg = [a.strip() for a in arg12.split(',')]

    ### mutuall exclusive things 
    if event1 != event2:
        ### drop and grab
        if ((event1 == "drop" and event2 == "grab") or (event1 == "grab" and event2 == "drop")):
            #### first part 
            if arg11 == arg12: arg_template = arg11
            elif g1[0] == g2[0]: arg_template = f"A {g1[0]}"
            else: arg_template = f"A person"

            ### final part 
            if arg21 == arg22: full = f"{arg_template} did something with the {arg21}"
            elif arg21 != arg22: full = f"{arg_template} did something with some object"
            else: raise ValueError('non-exhaustive grab/drop')
            return full

        ### grab and give 
        elif (event1 == "give" and event2 == "grab") or (event1 == "grab" and event2 == "give"):
            new_arg1 = first_arg[1] if event1 == "give" else first_arg[0]
            new_arg2 = second_arg[0] if event1 == "give" else second_arg[1]
            ## first part 
            if new_arg1 == new_arg2: arg_template = f"{new_arg1}"
            elif PEOPLE[new_arg1] == PEOPLE[new_arg2]: arg_template = f"A {PEOPLE[new_arg1]}"
            else: arg_template = "A person"
            
            ### final part 
            if arg21 == arg22: full = f"{arg_template} got the {arg21}"
            elif arg21 != arg22: full = f"{arg_template} got some object"
            else: raise ValueError('non-exhasutive give/grab')

            return full

        ### give and drop 
        elif (event1 == "give" and event2 == "drop") or (event1 == "drop" and event2 == "give"):
            new_arg1 = first_arg[0]
            new_arg2 = second_arg[0]
            if new_arg1 == new_arg2: arg_template = f"{new_arg1}"
            elif PEOPLE[new_arg1] == PEOPLE[new_arg2]: arg_template = f"A {PEOPLE[new_arg1]}"
            else: arg_template = "A person"

            if arg21 == arg22: full = f"{arg_template} lost the {arg21}"
            elif arg21 != arg22: full = f"{arg_template} lost some object"
            else: raise ValueError('non-ehasuctive give/drop')

            return full

        elif (event1 == "move" and event2 in set(["grab","drop","give"])) or (event1 in set(["grab","drop","give"]) and event2 == "move"):
            return ""
            # if arg11 == arg12:
            #     return f"Incompatible situation involving {arg11}"
            # return "Incompatible situation"
        else:
            raise ValueError(f'Unknown combination: {rep1}, {rep2}')

    ### arg1 template
    if arg11 == arg12:
        arg1_template = arg11.replace("_"," and ")
    elif arg11 != arg12 and event1 != 'give':
        if len(g1) == 1 and len(g2) == 1 and g1[0] == g2[0]: arg1_template = f"A {g1[0]}"
        elif len(g1) == 1 and len(g2) == 1 and g1[0] != g2[0]: arg1_template = f"A person"
        elif (len(g2) > 1 and len(g1) == 1) or (len(g1) > 1 and len(g2) == 1): arg1_template = f"One or more people"
        elif len(g1) > 1 and len(g2) > 1 and len(set(g1)) == 1 and set(g2) == set(g1): arg1_template = f"Two {g1[0]}s"
        elif len(g1) > 1 and len(g2) > 1 and len(set(g1)) == 2 and len(set(g2)) == 2: arg1_template = f"A female and male"
        elif len(g1) > 1 and len(g2) > 1: arg1_template = "Two people"

        else:
            raise ValueError(
                f'Unknown arg1 abstraction: {rep1},{rep2}'
            )
    elif event1 == "give":
        assert len(first_arg) == len(second_arg) == 2
        if first_arg[0] == second_arg[0] and first_arg[1] == second_arg[1]:
            arg1_template = f"{first_arg[0]} gave {first_arg[0]}"
        elif PEOPLE[first_arg[0]] == PEOPLE[second_arg[0]] and first_arg[1] == second_arg[1]:
            arg1_template = f"A {PEOPLE[first_arg[0]]} gave {first_arg[1]}"
        elif PEOPLE[first_arg[1]] == PEOPLE[second_arg[1]] and first_arg[0] == second_arg[0]:
            arg1_template = f"{first_arg[0]} gave a {PEOPLE[first_arg[1]]}"
        elif PEOPLE[first_arg[1]] != PEOPLE[second_arg[1]] and first_arg[0] == second_arg[0]:
            arg1_template = f"{first_arg[0]} gave someone"
        elif PEOPLE[first_arg[0]] != PEOPLE[second_arg[0]] and first_arg[1] == second_arg[1]:
            arg1_template = f"Someone gave {first_arg[1]}"
        elif PEOPLE[first_arg[0]] != PEOPLE[second_arg[0]] and PEOPLE[first_arg[1]] != PEOPLE[second_arg[1]]:
            arg1_template = f"Someone gave someone else"
        elif PEOPLE[first_arg[0]] == PEOPLE[second_arg[0]] and PEOPLE[first_arg[1]] != PEOPLE[second_arg[1]]:
            arg1_template = f"A {PEOPLE[first_arg[0]]} gave someone"
        elif PEOPLE[first_arg[1]] == PEOPLE[second_arg[1]] and PEOPLE[first_arg[0]] != PEOPLE[second_arg[0]]:
            arg1_template = f"Someone gave a {PEOPLE[first_arg[1]]}"
        elif PEOPLE[first_arg[1]] == PEOPLE[second_arg[1]] and PEOPLE[first_arg[0]] == PEOPLE[second_arg[0]]:
            arg1_template = f"A {PEOPLE[first_arg[0]]} gave a {PEOPLE[first_arg[1]]}"
        else:
            raise ValueError(f"unknown aggregation for reps: {rep1},{rep2}")
        
    else:
        raise ValueError(f"unknown event: {event1}")
    
    ### verb template
    if event1 == "move" and arg21 == arg22:   verb_template = f"{arg1_template} moved to the {arg21}"
    elif event1 == "move" and arg21 != arg22: verb_template = f"{arg1_template} moved somewhere"
    elif event1 == "drop" and arg21 == arg22: verb_template = f"{arg1_template} dropped the {arg21}"
    elif event1 == "drop" and arg21 != arg22: verb_template = f"{arg1_template} dropped something"
    elif event1 == "grab" and arg21 == arg22: verb_template = f"{arg1_template} grabbed the {arg21}"
    elif event1 == "grab" and arg21 != arg22: verb_template = f"{arg1_template} grabbed something"

    elif event1 == "give":
        if arg21 == arg22: verb_template = f"{arg1_template} the {arg21}"
        else: verb_template = f"{arg1_template} some object"
    else: raise ValueError(f'Unknown final rep pair: {rep1},{rep2}')
    return verb_template 

### CLUTTR utilties     

genders = {
    "brother"        : "male",
    "sister"         : "female",
    "grandfather"    : "male",
    "grandmother"    : "female",
    "granddaughter"  : "female",
    "grandson"       : "male",
    "mother"         : "female",
    "father"         : "male",
    "daughter"       : "female",
    "son"            : "male",
    "uncle"          : "male",
    "aunt"           : "female",
    "wife"           : "female",
    "husband"        : "male",
    "niece"          : "female",
    "nephew"         : "male",
    "father-in-law"  : "male",
    "mother-in-law"  : "female",
    "sister-in-law"  : "female",
    "brother-in-law"  : "male",
    "mother-in-law"  : "female",
    "father-in-law"  : "male",
    "son-in-law"      : "male",
    "daughter-in-law"      : "female",
}
det = {
    "brother"        : "a",
    "sister"         : "a",
    "grandfather"    : "a",
    "grandmother"    : "a",
    "granddaughter"  : "a",
    "grandson"       : "a",
    "mother"         : "a",
    "father"         : "a",
    "daughter"       : "a",
    "son"            : "a",
    "uncle"          : "an",
    "aunt"           : "an",
    "wife"           : "a",
    "husband"        : "a",
}


FEMALE = set([
    "Olivia",
    "Emma",
    "Ava",
    "Charlotte",
    "Sophia",
    "Amelia",
    "Isabella",
    "Mia",
    "Evelyn",
    "Harper",
    "Camila",
    "Gianna",
    "Abigail",
    "Luna",
    "Ella",
    "Avery",
    "Layla",
    "Penelope",
    "Nora",
    "Grace",
    "Zoey",
    "Stella",
    "Eliana",
    "Skylar",
    "Josephine",
    "Felicity",
    "Yulia",
    "Miroslava",
    "Margaret",
    "Hilary",
    "Alina",
    "Anastasia",
    "Hannah","Karen",
    "Patricia","Edith","Silvia","Debbie","Tina","Brandi","Kelly","Pearl","Sarah","Leslie","Lillian","Lois","Linda","Elizabeth","Pamela","Laura","Susan","Ernestine","Pauline","Mary","Gayle","Yolanda","Katie","Amber","Amada","Karen","Jennifer","Jacqueline","Dorothy","April","Lisa","Caroline","Helen","Denise","Vickie","Robin","Celesta","Merle","Pandora","Elma","Edyth","Nancy","Wendy","Marjorie","Ana","Qiana","Jenny","Maryellen","Beverly","Sara","Marie","Barbara","Sandra","Cindy","Lillie","Asuncion","Adelaide","Bonnie","Ruth","Ida","Teresa","Samantha","Sherryl","Heather","Ashley","Deborah","Elisa","Maricela","Carol","Elyse","Kathrine","Ida","Inge","Loretta","Tonja","Laveta","Brenda","Nichole","Jackie","Leigh","Alma","Juanita","Tabitha","Shirley","Alice","Phyllis","Jewel","Judith","Daria","Nathalie","Ethel","Dawn","Stephanie",
    "Rose","Katharine","Mellie","Rachel","Geraldine","Marguerite","Marguerite","Frances","Angela","Tanya","Marcia","Shirl","Natalie","Leila","Diane","Michelle","Carolyn","Dianna",
    "Lucy",
    "Liliana",
    "Ewelina",
    "Patricia",
    "Ariana",
    "Caroline",
    "Diana",
    "Claire",
    "Sylvia",
    "Irina",
    "Sonia",
    "Crystal",
    "Deirdre",
    "Marilyn",
    "Nikole",
])
MALE = set([
    "Liam",
    #"Noah",
    "Oliver","Elijah","William","James","Benjamin",
    #"Lucas",
    "Ethan","Sebastian","Jack","Owen","Theodore","Leo","Ezra","Nolan",
    "Hunter","Dominic","Wesley","Fedor","Vladimir","Bogdan","Gregor","Abraham","Elad","Oren","Frederick","Roberta","Niall","Christian","Roman","Marvin","Duance","David","John","Dean","Jon","Jeffrey",
    "Frankie","Victor","Bruce","Wilbert","Frank",
    "Robert", "Johnny", "Shane", "Fernando","Rogelio", "Kurt", "Edward", "Dominique","David","Mark","Lucas",
    "Miguel","Peter","Jeffrey","Eugene","Lawrence","Juan","Marc","Charlie","Charles","Rodney","Eric","Jake","Desmond",
    "Milton","Thomas","Harold","Devon","Philip","Russell","Alphonso","Otis","Michael","Steve","Richard","Steven","Bob",
    "Francis","Francisco","Stephen","Derrick","Seth","Raymond","Jesus","Jerry","James","Kraig","Chester","Mario","Harry",
    "Larry","Fred","Caleb","Tom","Myron","John","Marcus","Jesse",
    "Chris","Justin","Samuel","Andrew","Craig","Dennis","Melvin","Gene","Randal","Duane","Matthew","George","Irving","Walter",
    "Paul","Jonathan","Bart","Rick","Kevin","Tim","Don","Romeo","Joseph","Jim","Travis","Leon","Troy","Daniel","Brian","Daniel",
    "Dustin","Jason","Anthony","Jose","Archie","Jonathon","Dwayne","Lonnie","Luis","Salome",
    "Gerald","Albert","Alfred","Anders","Axel","Felix","Jasper","Karl","Ivan","Hector","Hugo","Conor","Carlos","Goran","Maurice","Dong","Timothy","Kendrick","Gregory","Bryant","Demetrius","Jaime","Foster"
])

GENDERS = {
    n:"female" for n in FEMALE
}
GENDERS.update({
    n:"male" for n in MALE
})

INVERSE = {
    ("male","daughter")   : "parent",
    ("female","daughter") : "parent",
    ("male","son")   : "parent",
    ("female","son") : "parent",
    ("male","granddaughter")   : "grandparent",
    ("female","granddaughter") : "grandprarent",
    ("male","grandson")   : "grandparent",
    ("female","grandson") : "grandparent",
    ("female","sister")   : "sibling",
    ("male","sister")     : "sibling",
    ("female","brother")  : "sibling",
    ("male","brother")    : "sibling",
    ("female","mother")   : "child",
    ("male","mother")     : "child",
    ("female","father")   : "child",
    ("male","father")     : "child",
    ("male","wife")       : "spouse",
    ("female","wife")       : "spouse",
    ("female","husband")       : "spouse",
    ("female","uncle"): "niece",
    ("male","uncle"): "nephew",
    ("female","aunt"): "niece",
    ("male","aunt"): "nephew",
    ("male","grandmother"): "grandchild",
    ("female","grandmother"): "grandchild",
    ("male","grandfather"): "grandchild",
    ("female","grandfather"): "grandchild",
    ("male","husband"): "spouse",    
}

ABSTRACTION = {
    "son"           : "child",
    "daughter"      : "child",
    "mother"        : "parent",
    "father"        : "parent",
    "grandmother"   : "grandparent",
    "grandfather"   : "grandparent",
    "brother"       : "sibling",
    "sister"        : "sibling",
    "uncle"         : "uncle",
    "aunt"          : "aunt",
    "nephew"        : "nephew",
    "niece"         : "niece",
    "wife"          : "spouse",
    "husband"       : "spouse",
    "grandson"      : "grandchild",
    "granddaughter" : "grandchild",
}

def compute_cluttr_abstraction(rep1,rep2):
    raw_arg11,rel1,raw_arg12 = rep1.split("--")
    raw_arg21,rel2,raw_arg22 = rep2.split("--")
    arg11_name,arg11_gender  = raw_arg11.split("_")
    arg12_name,arg12_gender  = raw_arg12.split("_")
    arg21_name,arg21_gender  = raw_arg21.split("_")
    arg22_name,arg22_gender  = raw_arg22.split("_")
    final = ""


    ### abstract arg1
    if rel1 == rel2:
        rel = rel1

            ### unknown
        if arg11_gender == "unk" or arg22_gender == "unk":
            return final
        elif arg11_name == arg21_name:
            role = INVERSE[(arg11_gender,rel)]
            arg1_abstract = f"{arg11_name} has a {genders[rel]} {rel}"
        elif arg11_gender == arg21_gender:
            role = INVERSE[(arg11_gender,rel)]
            arg1_abstract = f"A {arg11_gender} person has a {genders[rel]} {rel}"
        else:
            role = INVERSE[("male",rel)]
            arg1_abstract = f"A male or female person has a {genders[rel]} {rel}"

        ### arg2 abstract
        arg2_abstract = ""
        if arg12_name == arg22_name: arg2_abstract = f"named {arg12_name}"
        final = f"{arg1_abstract} {arg2_abstract}"

    elif (rel1 == "sister" and rel2 == "brother") or (rel2 == "sister" and rel1 == "brother"):
        if arg11_name == arg21_name: final =  f"{arg11_name} has a sibling"
        elif arg11_gender == arg21_gender: final = f"A {arg11_gender} person has a sibling"
        else:  final = f"A male or female person has a sibling"
    elif (rel1 in "husband" and rel2 == "wife") or (rel2 == "husband" and rel1 == "wife"):
        final = "A male or female person has a spouse"
    elif (rel1 == "father" and rel2 == "mother") or (rel2 == "father" and rel1 == "mother"):
        if arg11_name == arg21_name: final = f"{arg11_name} has a parent"
        elif arg11_gender == arg21_gender: final = f"A {arg11_gender} person has a parent"
        else: final = f"A male or female person has a parent"
    elif (rel1 == "grandmother" and rel2 == "grandfather") or (rel2 == "grandmother" and rel1 == "grandfather"):
        if arg11_name == arg21_name: final = f"{arg11_name} has a grandparent"        
        if arg11_gender == arg21_gender: final = f"A {arg11_gender} person has a grandparent"
        else: final = f"A male or female person has a grandparent"
    elif (rel1 == "daughter" and rel2 == "son") or (rel2 == "daughter" and rel1 == "son"):
        if arg11_name == arg21_name: final = f"{arg11_name} has a child"        
        if arg11_gender == arg21_gender: final = f"A {arg11_gender} person has a child"
        else: final = f"A male or female person has a child"
    elif (rel1 == "granddaughter" and rel2 == "grandson") or (rel2 == "granddaughter" and rel1 == "grandson"):
        if arg11_name == arg21_name: final = f"{arg11_name} has a grandchild"        
        if arg11_gender == arg21_gender: final = f"A {arg11_gender} person has a grandchild"
        else: final = f"A male or female person has a grandchild"
    elif (rel1 == "aunt" and rel2 == "uncle") or (rel2 == "aunt" and rel1 == "uncle"):
        if arg11_name == arg21_name: final = f"{arg11_name} has an aunt or uncle"        
        if arg11_gender == arg21_gender: final = f"A {arg11_gender} person has an aunt or uncle"
        else: final = f"A male or female person has an aunt or uncle"

    return final
