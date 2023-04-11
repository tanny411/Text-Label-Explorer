Please click [here](https://tanny411.github.io/Text-Label-Explorer/) to view the project.

## Student Information
- Name: Aisha Khatun
- Student ID: 21012959
- WhatIAM: a2khatun

## Installation

If you want to run the python scripts, install the python requirements.
```bash
pip3 install -r requirements.txt
```

## Usage

### Python script
The python script is already run and data stored in the appropriate place. If you want to run with your own data, save your data as `data/data.csv` following the format of the current data.csv file. Then run the python scripts:

```bash
python3 pyscripts/generate_embeddings.py
python3 pyscripts/get_topics.py
python3 pyscripts/arrange_data.py
```
The data will be stored in `data` and `pubilc` folder.

### Setting up the server
Inder to successfully load the resources, you need to use a Node.js server or a simple python server.

#### Node.js server
* The server app `final_app.js` has already been provided.
* Initialize a new Node.js project by running the command:
    `npm init`  
* Install Express as a dependency for your project by running the command:
      `npm install express`
  This will install the latest version of Express and add it to your project's `package.json` file.

* Run the server with the command `npm start`  or `node final_app.js`
This will start the Node.js server with Express.

#### Python simple server
* To use python simple server, move all the files to the same location or folder. i.e all the datasets (csv files) and the style.css, `d3.layout.cloud.js`, and `index.html`
* Open `index.html` and edit the file paths for all the referenced files. Since the files are now located in the same directory as `index.html`, remove the prefixes. E.g
Change the reference to topic files from `d3.csv(`text_label_explorer/${topic_file}.csv`... to `d3.csv(`${topic_file}.csv`,function(data)`
The required changes are:
** `href="public/style.css"` to `href="style.css"`
** `d3.csv("public/embeddings_new.csv"...`   to `d3.csv("embeddings_new.csv"...`
** `d3.csv("public/top_5_matches.csv"...`    to `d3.csv("top_5_matches.csv"...`
** `d3.csv(`public/top_topics/${techniqueColumn}.csv`....` to `d3.csv(`${techniqueColumn}.csv`...`
** `d3.csv(`public/text_label_explorer/${topic_file}.csv`..` to `d3.csv(`public/${topic_file}.csv`..` to `d3.csv(`${topic_file}.csv`...`

* Lastly, run `python -m http.server` to start python simple server
* Navigate to `http://localhost:8080` in your web browser
* locate the index.html file, then click to open.




### View project

You can view the UI by opening `index.html` in your browser.

## License

This project is open source and available under the [MIT License](LICENSE).
