from pydantic import BaseModel
import json
import re


import logging
logger = logging.getLogger(__name__)

# Output formatting for LLM-based rubric evaluation
class RubricResponse(BaseModel):
    reason: str
    score: float
    
    @classmethod
    def parse_raw(cls, output):
        # a json output
        if output.startswith("{"): 
            try:
                # Try to complete the json string if incomplete due to max_out_length
                output = output.strip()
                if not (output.endswith("}")):
                    output += "}" 
                parsed_json = json.loads(output)
                # Check if the parsed json is in the exact format that we want
                assert 'score' in parsed_json and 'reason' in parsed_json
                # Score must be able to convert into a number 
                parsed_json['score'] = float(parsed_json['score']) 

                parsed_response = RubricResponse(**parsed_json)
                return parsed_response        
            except:
                logger.warning(f"Error in parsing with JSON with appended curly bracket:\n{output}")

            # Also try completing json with a "}
            try:
                output = output.strip()
                if not (output.endswith("\"}")):
                    output += "\"}" 
                parsed_json = json.loads(output)

                assert 'score' in parsed_json and 'reason' in parsed_json
                # Score must be able to convert into a number 
                parsed_json['score'] = float(parsed_json['score']) 

                parsed_response = RubricResponse(**parsed_json)
                return parsed_response        
            except:
                logger.warning(f"Error in parsing with JSON when appending quote+curly bracket:\n{output}")
                

        # Normalize whitespace
        output = re.sub(r'\s+', ' ', output).strip()
        
        match = re.match(r'(.+?)\b(?:score|scoring|rate|rating|answer)\b(?:[^0-9]*)\b(\d+)', output, re.IGNORECASE) 
        if match:
            try:
                return RubricResponse(score=float(match.group(2)), reason=match.group(1).strip())
            except:
                logger.warning(f"Error in parsing using regex:\n{[output]}")
                return RubricResponse(score=-1, reason=output)
        else:
            logger.warning(f"Error in parsing, no regex match found:\n{[output]}")
            return RubricResponse(score=-1, reason=output)