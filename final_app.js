const express = require('express');
const app = express();
const http = require('http').Server(app);
// const io = require('socket.io')(http);
// const d3 = require('d3');

app.use(express.static('public'));

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/last_iter.html');
});

http.listen(3500, () => {
  console.log('Listening on port 3500');
});