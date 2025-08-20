// @ts-nocheck
import { SpeechService } from "$lib/services/speech-service";

export async function POST({ request }) {
  const { audio } = await request.json();
  const transcript = await SpeechService.transcribe(audio);
  return json({ transcript });
}
