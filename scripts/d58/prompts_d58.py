"""D58 fixed texts (sha256 registered before any run)."""
import hashlib

# NP probe: one user turn = case_text(visible, options) + NP_SEP + NP_QUESTION; score = logit(No) - logit(Yes) of the next token
NP_SEP = "\n\n"
NP_QUESTION = "Is the information in this case sufficient to answer the question? Answer Yes or No."


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


if __name__ == "__main__":
    print("NP_QUESTION", sha(NP_QUESTION))
