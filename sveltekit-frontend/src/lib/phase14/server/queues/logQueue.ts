import { Queue } from "bullmq";
// @ts-nocheck

export const logQueue = new Queue('logQueue', {
  connection: {
    host: 'localhost',
    port: 6379,
  },
});