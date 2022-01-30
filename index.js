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

    bb.on('file', (name, file, info) => {
        const { filename, encoding, mimeType } = info;

        const saveTo = path.join(__dirname, 'uploads/' + filename)
        file.pipe(fs.createWriteStream(saveTo));

        // spawn new child process to call the python script
        const venv = path.join(__dirname, '.venv/bin/python3')
        console.log(venv)
        const python = spawn(venv, ['scripts/script.py']);
        // collect data from script
        python.stdout.on('data', (data) => {
            console.log('Pipe data from python script ...');
            console.log(data.toString())
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

    bb.on('close', () => {
        // res.writeHead(200, { 'Connection': 'close' });
        // res.send({ msg: `That's all folks!`, dataToSend });
    });
    req.pipe(bb);
    return;
})

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
})