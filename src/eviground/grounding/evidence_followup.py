"""An evidence gap with a concrete acquisition that can address it."""


class EvidenceFollowup(ValueError):
    def __init__(self, message, *, focus_ids, tool='measure', frame=None, condition_id=None):
        super().__init__(message)
        self.focus_ids = list(dict.fromkeys(focus_ids))
        self.tool = tool
        self.frame = frame
        self.condition_id = condition_id

    def details(self):
        return {'tool': self.tool, 'focus_ids': self.focus_ids, 'frame': self.frame,
                'condition_id': self.condition_id}


def was_attempted(details, observations):
    return any(row.get('evidence_followup') == details for row in observations)
