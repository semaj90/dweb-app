// WebSocket server for real-time collaboration
import { WebSocketServer } from "ws";
import { createServer } from "http";

const server = createServer();
const wss = new WebSocketServer({ server });

const rooms = new Map();

wss.on("connection", (ws) => {
  console.log("Client connected");

  ws.on("message", (data) => {
    try {
      const message = JSON.parse(data.toString());

      switch (message.type) {
        case "JOIN_ROOM":
          if (!rooms.has(message.room)) {
            rooms.set(message.room, new Set());
          }
          rooms.get(message.room).add(ws);
          ws.currentRoom = message.room;
          break;

        case "NODE_CREATED":
        case "NODE_UPDATED":
          // Broadcast to room members
          if (ws.currentRoom && rooms.has(ws.currentRoom)) {
            rooms.get(ws.currentRoom).forEach((client) => {
              if (client !== ws && client.readyState === 1) {
                client.send(JSON.stringify(message));
              }
            });
          }
          break;
      }
    } catch (error) {
      console.error("WebSocket message error:", error);
    }
  });

  ws.on("close", () => {
    // Remove from room
    if (ws.currentRoom && rooms.has(ws.currentRoom)) {
      rooms.get(ws.currentRoom).delete(ws);
    }
    console.log("Client disconnected");
  });
});

const PORT = process.env.WS_PORT || 8080;
server.listen(PORT, () => {
  console.log(`WebSocket server running on port ${PORT}`);
});
