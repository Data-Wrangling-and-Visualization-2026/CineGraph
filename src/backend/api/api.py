from fastapi import FastAPI

app = FastAPI()

@app.get('/')
async def general():
    return {
        'status': 'accepted'
    }

