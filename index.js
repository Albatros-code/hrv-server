const busboy = require('busboy');
const path = require('path');
const fs = require('fs');
const express = require('express');
const { spawn } = require('child_process');

const app = express()
const port = 5000

app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "http://localhost:3000");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});

app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use(express.static("build"))

app.get('/hello-world', (req, res) => {
    res.send('Hello World!')
})

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "build", "index.html"));
});

app.post('/calculate-hrv', (req, res) => {

    const bb = busboy({ headers: req.headers })

    let dataToSend
    let parameters
    let saveTo

    bb.on('field', (name, val, info) => {
        console.log(`Field [${name}]: value: %j`, JSON.parse(val))
        parameters = JSON.parse(val)
    });
    bb.on('file', (name, file, info) => {
        const { filename, encoding, mimeType } = info;

        saveTo = path.join(__dirname, 'uploads/' + filename)
        const saveFile = fs.createWriteStream(saveTo)
        file.pipe(saveFile);

        saveFile.on('close', () => {
            console.log('File saved')

        })

    });

    bb.on('close', () => {
        // spawn new child process to call the python script
        const venv = path.join(__dirname, '.venv/bin/python3')
        const prodVenv = 'python'
        console.log('Starting python script')
        const python = spawn(process.env.NODE_ENV === 'development' ? venv : prodVenv, ['scripts/script.py', saveTo, "--step", parameters.step, "--window", parameters.window]);
        // collect data from script
        python.stdout.on('data', (data) => {
            console.log('Pipe data from python script ...');
            dataToSend = data.toString();
        });

        python.stderr.on('data', (data) => {
            console.log(data.toString())
        });
        // in close event we are sure that stream from child process is closed
        python.on('close', (code) => {
            console.log(`child process close all stdio with code ${code}`);
            res.send(dataToSend);
            fs.rm(saveTo, () => { console.log("Removing" + saveTo) })
            // send data to browser
        });
    });
    req.pipe(bb);
    return;
})

app.listen(process.env.PORT || port, () => {
    console.log(`Example app listening on port ${port}`)
})
