import os
import pickle
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# Load the machine learning model
model = pickle.load(open("D:/template/svm_clf.pkl", "rb"))

# Set the template folder path
template_folder = os.path.join("D:/", "template", "templates")
templates = Jinja2Templates(directory=template_folder)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class Item(BaseModel):
    # Define your request body model here
    ...

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(item: Item):
    # Your prediction logic here
    ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
