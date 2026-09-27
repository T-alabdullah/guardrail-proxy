"""Checks that the official OpenAI client works against the proxy with only base_url changed."""
import openai
from openai import OpenAI

# The only change from talking to OpenAI: base_url.
# The client insists on *some* api_key; the proxy ignores it (for now).
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

print("openai client version:", openai.__version__)

# 1. Normal request
resp = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Why is the sky blue? One sentence."}],
)
print("\n[1] content:", resp.choices[0].message.content)
print("    finish_reason:", resp.choices[0].finish_reason)
print("    usage:", resp.usage)

# 2. max_tokens should cut the answer short -> finish_reason 'length'
resp = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Tell me a long story about a dragon."}],
    max_tokens=10,
)
print("\n[2] finish_reason (expect 'length'):", resp.choices[0].finish_reason)

# 3. Unknown model should raise the client's NotFoundError
try:
    client.chat.completions.create(
        model="does-not-exist",
        messages=[{"role": "user", "content": "hi"}],
    )
    print("\n[3] ERROR: expected an exception")
except openai.NotFoundError as e:
    print("\n[3] got NotFoundError as expected:", e.message)

# 4. Model listing
print("\n[4] models:", [m.id for m in client.models.list()])
