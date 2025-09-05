#!/usr/bin/env node
import http from 'http'
import { WebSocketServer } from 'ws'

const PORT = Number(process.env.LOG_WS_PORT || 7071)

const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' })
  res.end('Log WS server running')
})

const wss = new WebSocketServer({ server, path: '/logs' })

function broadcast(obj) {
  const data = JSON.stringify(obj)
  for (const client of wss.clients) {
    if (client.readyState === 1) client.send(data)
  }
}

setInterval(() => {
  broadcast({ ts: new Date().toISOString(), level: 'info', msg: 'heartbeat' })
}, 5000)

server.listen(PORT, () => console.log(`WS logs at ws://localhost:${PORT}/logs`))
