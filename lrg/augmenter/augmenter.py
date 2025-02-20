from typing import List, Dict, Tuple, Literal
import json
from pydantic import BaseModel

import pandas as pd

from llama_index.core.schema import NodeWithScore
from ..data import EvalDataset

class NitiLinkAugmenterConfig(BaseModel):
    # Strat name here refers to the chunking strategy. The strategy can be any name as long as it is a string.
    # If it contains golden, that means it uses golden chunking strategy which means each chunk is a section and can be used to do link.
    # If it doesn't, it cannot be used to do cross ref. We chose to do this because for naive chunking, even though we know how each section links to each other, we couldn't know if the referencing span is in the chunked section or not!
    strat_name: str = "golden"
    reference: bool = False
    max_depth: int = 1
    
class NitiLinkAugmenter(object):
    """
    NitiLinkAugmenter class. Need to have the following function
    1. add_xml_tag: For when reference option is set to false or strat name is not golden
    2. link, get_relevant_laws, relevant_laws_to_str: For retrieveing all relevant laws
    3. __call__: main option for taking in query and retrieved node
    """
    
    def __init__(self, dataset: EvalDataset, config: NitiLinkAugmenterConfig):
        """
        Save config in its own attribute and take in dataset as well. Need to load section mapping as well
        """
        self.config = config.model_dump()
        self.dataset = dataset
        
    def add_xml_tag(self, node: NodeWithScore, strat_name: str):

        metadata = node.metadata
        text = node.text

        if "golden" not in strat_name:
            covered_sections = metadata["sections_covered"]

            #Split text into sections, then join each other with xml tag
            indices = self.dataset.section_idx[metadata["law_name"]].loc[covered_sections].to_dict()

            split_sections = []

            for c in covered_sections:
                split_sections.append(f"<law section={c} law_name={metadata['law_name']}> ")
                split_sections.append(text[max(indices[c][0] - metadata["start_index"], 0): min(indices[c][1] - metadata["start_index"], len(text))])
                split_sections.append(" </law>\n")

            return "".join(split_sections)
        else:
            law_name, section = node.id_.split("-")
            return f"<law section={section} law_name={law_name}> {text} </law>"
        
    def link(self, main_law: Tuple[str, str], relevant_laws: List = [], curr_depth: int = 0, max_depth: int = 2):
        
        law_name_dict = self.dataset.law_name_dict
        
        if (curr_depth > max_depth) or (main_law in relevant_laws):
            return relevant_laws
        #Also check if the main law exists, if not, just return the current set
        if not main_law[1] in self.dataset.sections[law_name_dict[main_law[0]]].index:
            return relevant_laws
        else:
            relevant_laws.append(main_law)
            #Then, drill down on the main law
            references = self.dataset.sections[law_name_dict[main_law[0]]].loc[main_law[1]]["reference"]
            for reference in references:
                if not reference["law_name"] in law_name_dict:
                    continue

                reference = (reference["law_name"], reference["section_num"])
                if not reference in relevant_laws:
                    relevant_laws = self.link(reference, relevant_laws, curr_depth + 1, max_depth)

            return relevant_laws

    def get_relevant_laws(self, laws: List[Tuple[str, str]], max_depth: int = 2):

        added_law = set()
        return_law = dict()

        for r in laws:
            curr_laws = self.link((r[0], r[1]), relevant_laws = [], curr_depth = 0, max_depth = max_depth)
            return_law[(r[0], r[1])] = []

            for l in curr_laws:
                if l not in added_law:
                    return_law[(r[0], r[1])].append(l)
                    added_law.add(l)

        #Sanity check, no duplicates
        assert len(added_law) == len(set(sum(return_law.values(), []))), "Somethings wrong, got len(return_law) more than len(set(return_law))"

        return return_law

    def relevant_laws_to_str(self, relevant_laws: Dict[Tuple[str, str], List[Tuple[str, str]]]):
        
        law_name_dict = self.dataset.law_name_dict
        str_list = []
        for k in relevant_laws:
            str_list.append(f"<law section={k[1]} law_name={k[0]}> {self.dataset.sections[law_name_dict[k[0]]].loc[k[1]]['section_content']} </law>")
            for r in relevant_laws[k][1:]:
                str_list.append(f"<related_law section={r[1]} law_name={r[0]} parent_section={k[1]} parent_law_name={k[0]}> {self.dataset.sections[law_name_dict[r[0]]].loc[r[1]]['section_content']} </related_law>")

        return "\n".join(str_list)

    def __call__(self, query: str, retrieved_nodes: List[NodeWithScore] = None):
        """
        Main function for augmenting query.
        If strat is other than golden. Only call add_xml_tag
        If strat is golden and reference is True, get_relevant_laws. If not call add_xml_tag
        """
        if retrieved_nodes is not None:
            new_query = ["<ข้อกฎหมาย>\n"]

            if "golden" not in self.config["strat_name"]:
                new_query = new_query + [self.add_xml_tag(node, strat_name=self.config["strat_name"]) + "\n" for node in retrieved_nodes]
            else:
                if self.config["reference"]:
                    retrieved_laws = [(x.id_.split("-")[0], x.id_.split("-")[1]) for x in retrieved_nodes]
                    relevant_laws = self.get_relevant_laws(retrieved_laws, self.config["max_depth"])
                    relevant_law_str = self.relevant_laws_to_str(relevant_laws)
                    new_query.append(relevant_law_str + "\n")

                else:
                    new_query = new_query + [self.add_xml_tag(node, strat_name=self.config["strat_name"]) + "\n" for node in retrieved_nodes]

            new_query += ["</ข้อกฎหมาย>\n", query]
            new_query = "".join(new_query)

            return new_query
        
        else:
            return query