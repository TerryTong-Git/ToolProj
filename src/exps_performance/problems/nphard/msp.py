from problems.nphardeval import NPHardEvalProblem
from prompts import mspPrompts


class MSP(NPHardEvalProblem):
    def __init__(self):
        self.p = mspPrompts

    def format_one(self, q) -> str:
        participants = q["participants"]
        prompt_text = (
            self.instantiate_prompt(dict(total_participants=participants, total_timeslots=q["time_slots"]))
            + "\n The meetings and participants details are as below: \n"
        )

        for meeting in q["meetings"]:
            this_line = "Meeting {} is with duration {}.".format(meeting["id"], meeting["duration"])
            prompt_text += this_line + "\n"
        for j in participants.keys():
            this_line = "Participant {} is available at time slots {} and has meetings {}.".format(
                j, participants[j]["available_slots"], participants[j]["meetings"]
            )
            prompt_text += this_line + "\n"
        return prompt_text
