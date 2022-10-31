## simple parsers for constraitns (currently very limited)
from typing import Dict, List
import sys
from pathlib import Path
import json
import re
import itertools
import logging

from typing import Union, List, Set, Optional
from dataclasses import dataclass, field, asdict
from rply import LexerGenerator,ParserGenerator


util_logger = logging.getLogger('situation_modeling.readers.constraints')

### CONSTRAINT INTERPRETER
lg = LexerGenerator()

lg.add('IMPLIES',r'Implies')
lg.add('AND',r'And')
lg.add('OR',r'Or')
lg.add('NEG',r'Negate')
lg.add('OPEN_PARENS', r'\(')
lg.add('CLOSE_PARENS', r'\)')
lg.add('UNDERSCORE', r'\_')
lg.add('NUMBER', r'\d+')
lg.add('COMMA', r'\,')
lg.add('BICONDITIONAL',r'Biconditional')
lg.add('DESCRIPTION',r'[a-zA-Z\-]+')

lg.ignore('\s+')

lexer = lg.build()

pg = ParserGenerator(
    [
        'IMPLIES',
        'AND',
        'OPEN_PARENS',
        'CLOSE_PARENS',
        'UNDERSCORE',
        'NUMBER',
        'COMMA',
        'DESCRIPTION',
        'NEG',
        'BICONDITIONAL',
        'OR',
    ],
)

class Operator(object):
    @property
    def operator_type(self):
        raise NotImplementedError
    @property 
    def antecedent(self):
        raise NotImplementedError
    @property 
    def consequent(self):
        raise NotImplementedError
    @property
    def rule_name(self):
        return self.name

class Conjunction(Operator):
    def __init__(self,prop,name='rule'):
        self.prop_list = [prop]
        self.name = name
    def add(self,new_prop):
        self.prop_list.append(new_prop)
    def __str__(self):
        if len(self.prop_list) == 1:
            return f"{self.prop_list[0]}"
        p_str = ','.join([str(p) for p in self.prop_list])
        return f"And({p_str})"

    def props(self):
        return [str(s) for s in self.prop_list]
    @property
    def operator_type(self):
        return "conjunction"
    @property 
    def antecedent(self):
        return self.props()
    @property 
    def consequent(self):
        return []
    
class Disjunction(Operator):
    def __init__(self,prop,name='rule'):
        self.prop_list = [prop]
        self.name = name
    def add(self,new_prop):
        self.prop_list.append(new_prop)
    def __str__(self):
        if len(self.prop_list) == 1:
            return f"{self.prop_list[0]}"
        p_str = ','.join([str(p) for p in self.prop_list])
        return f"Or({p_str})"
    def props(self):
        return [str(s) for s in self.prop_list]

    @property
    def operator_type(self):
        return "disjunction"
    @property
    def operator_type(self):
        return "disjunction"
    @property 
    def antecedent(self):
        return self.props()
    @property 
    def consequent(self):
        return []

class Proposition(object):
    def __init__(self,left_id,right_id,polarity=1):
        self.left = left_id.value
        self.right = right_id.value if right_id is not None else None
        self.polarity = polarity
        
    def __str__(self):
        if self.right is None:
            if self.polarity == 0: 
                return f"~{self.left}"
            return f"{self.left}"
        if self.polarity == 0: 
            return f"~{self.left}_{self.right}"
        return f"{self.left}_{self.right}"

    def props(self):
        return [str(self)]

class Implication(Operator):

    def __init__(self,n,left,right):
        self.left = left
        self.right = right
        self.name = n

    @classmethod
    def initialize(cls,n='rule'):
        return lambda x : lambda y : cls(n,x,y)

    def __str__(self):
        return f"{self.name}_Implies({self.left},{self.right})"

    @property 
    def antecedent(self):
        return self.left.props()
    @property 
    def consequent(self):
        return self.right.props()
    @property
    def rule_name(self):
        return self.name

    @property
    def operator_type(self):
        return 'implication'

class Biconditional(Implication):
    @property
    def operator_type(self):
        return 'biconditional'

    def __str__(self):
        return f"{self.name}_Biconditional({self.left},{self.right})"

#@pg.production('expression : rule_header OPEN_PARENS expr COMMA expr CLOSE_PARENS')
@pg.production('expression : rule_header OPEN_PARENS expr COMMA expr CLOSE_PARENS')
@pg.production('expression : bin')
def implication_expression(p):
    if len(p) == 1:
        return p[0]
    return p[0](p[2][0])(p[4][0])

@pg.production('expr : bin')
@pg.production('expr : prop')
def expr_rule(p):
    return p

@pg.production('rule_header : DESCRIPTION UNDERSCORE IMPLIES')
@pg.production('rule_header : IMPLIES')
def header(p):
    if len(p) == 1:
        return Implication.initialize()
    return Implication.initialize(p[0].value)

@pg.production('rule_header : DESCRIPTION UNDERSCORE BICONDITIONAL')
@pg.production('rule_header : BICONDITIONAL')
def header_v2(p):
    if len(p) == 1:
        return Biconditional.initialize()
    return Biconditional.initialize(p[0].value)

@pg.production('bin : left_conj CLOSE_PARENS')
def conjunction1(p):
    return p[0]

@pg.production('prop : NEG OPEN_PARENS prop CLOSE_PARENS')
def basic_proposition(p):
    p[2].polarity = 0
    return p[2] 

@pg.production('prop : NUMBER UNDERSCORE NUMBER')
def basic_proposition(p):
    return Proposition(p[0],p[2])

@pg.production('prop : NUMBER')
def basic_proposition(p):
    return Proposition(p[0],None)

@pg.production('left_conj : left_conj COMMA prop')
def partial_conj_2(p):
    p[0].add(p[2])
    return p[0]

@pg.production('left_conj : AND OPEN_PARENS prop')
def partial_conj(p):
    conj = Conjunction(p[-1])
    return conj

@pg.production('left_conj : OR OPEN_PARENS prop')
def partial_disj(p):
    conj = Disjunction(p[-1])
    return conj

@pg.error
def error_handler(token):
    raise ValueError("Ran into a %s where it wasn't expected, token=%s" %\
                         (token.gettokentype(),token.value))

parser = pg.build()

@dataclass
class Constraint:
    name: str = field(init=True, repr=True, default='gen')
    arg1: List[str] = field(default_factory=list)
    arg2: List[str] = field(default_factory=list)
    batch_id: int = field(init=False, repr=False, default=-1)
    operator_type: str = field(init=True, repr=True, default='implication')

    @property
    def antecedent(self):
        """Returns a normalized version of the left side if implication (i.e., `arg1`)

        :rtype: list 
        """
        if self.batch_id != -1:
            return [f"{a}_{self.batch_id}" for a in self.arg1]
        return self.arg1

    @property
    def consequent(self):
        """Returns a normalized version of the right side if implication (i.e., `arg2`)

        :rtype: list 
        """
        if self.batch_id != -1:
            return [f"{a}_{self.batch_id}" for a in self.arg2]
        return self.arg2

    def left_right_values(self):
        """Translates to left and right rules 

        :rtype: tuple
        :returns: a tuple of the constraint name, operator type and parsed `left` and `right` components 
        """
        left = [
            [int(i) for i in idx.replace("~","").split("_")]+[0] if "~" in idx else\
            [int(i) for i in idx.replace("~","").split("_")]+[2] \
            for idx in self.antecedent
        ]
        right = [
            [int(i) for i in idx.replace("~","").split("_")]+[0] if "~" in idx else\
            [int(i) for i in idx.replace("~","").split("_")]+[2] \
            for idx in self.consequent
        ]
        return (self.name,self.operator_type,left,right)

    def cnf_expr(self):
        """Translates constraint into CNF format 

        :rtype: list 
        :returns: a list of clauses in tuple form
        """
        clause_list = [] 
        if self.operator_type == "implication":
            left = [
                (0,tuple([int(z) for z in l.replace("~","").split("_")+[2]])) if "~" not in l \
                else (1,tuple([int(z) for z in l.replace("~","").split("_")+[2]])) \
                for l in self.antecedent
            ]
            ## split up the disjunctions in consequent 
            for atom in self.consequent:
                raw = [int(z) for z in atom.replace("~","").split("_")+[2]]
                t = (0,tuple(raw)) if "~" in atom else (1,tuple(raw))
                clause_list.append(left+[t])
        elif self.operator_type == "disjunction":
            left = [
                (0,tuple([int(z) for z in l.replace("~","").split("_")+[2]])) if "~" in l \
                else (1,tuple([int(z) for z in l.replace("~","").split("_")+[2]])) \
                for l in self.antecedent
            ]
            clause_list.append(left)
        elif self.operator_type == "biconditional":
            ### first side
            left_first = [
                (0,tuple([int(z) for z in l.replace("~","").split("_")+[2]])) if "~" not in l \
                else (1,tuple([int(z) for z in l.replace("~","").split("_")+[2]])) \
                for l in self.antecedent
            ]
            for atom in self.consequent:
                raw = [int(z) for z in atom.replace("~","").split("_")+[2]]
                t = (0,tuple(raw)) if "~" in atom else (1,tuple(raw))
                clause_list.append(left_first+[t])
            ### reverse side
            left_second = [
                (0,tuple([int(z) for z in l.replace("~","").split("_")+[2]])) if "~" not in l \
                else (1,tuple([int(z) for z in l.replace("~","").split("_")+[2]])) \
                for l in self.consequent
            ]
            for atom in self.antecedent:
                raw = [int(z) for z in atom.replace("~","").split("_")+[2]]
                t = (0,tuple(raw)) if "~" in atom else (1,tuple(raw))
                clause_list.append(left_second+[t])

        else:
            raise ValueError(
                f'Unknown/un-implemented operator: {self.operator_type}'
            )
        return (self.name,clause_list)

### shold expand to a more general parser with more constraints 
def parse_basic_implications_conjunctions(constraint_labels,exclude=''):
    """Some initial pre-processing on the set of constraints 

    :param constraint_labels: the raw symbolic constraints 
    :type constraint_labels: list 
    :raises: ValueError 
    :rtype: list 
    """
    #constraint_list = []
    constraint_list = []
    excluded = set([r.strip() for r in exclude.split(';')])

    for constraint in constraint_labels:
        try:
            try: 
                parser_out = parser.parse(lexer.lex(constraint))
            except:
                try:
                    parser_out = parser.parse(lexer.lex(constraint+")"))
                except Exception as e:
                    raise e

        except Exception as e:
            util_logger.warning(
                f'Cannot parse constraint={constraint},error={e}',
                exc_info=True
            )
            continue

        first  = parser_out.antecedent
        second = parser_out.consequent
        cname  = parser_out.name

        constraint = Constraint(
            arg1=first,
            arg2=second,
            name=cname,
            operator_type=parser_out.operator_type
        )
        constraint_list.append(constraint)
        
    return constraint_list
