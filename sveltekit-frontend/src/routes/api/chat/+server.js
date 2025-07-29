import { env } from "$env/dynamic/private";
import { json } from "@sveltejs/kit";

/**
 * @type {import('./$types').RequestHandler}
 */
export async function POST({ request }) {
  try {
    const { messages } = await request.json();

    // Make a request to the Ollama service inside the Docker network
    const response = await fetch(`${env.OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gemma3:latest", // Available model for RTX 3060
        messages: messages,
        stream: true,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama API error: ${response.status} ${errorText}`);
    }

    // Create a ReadableStream to pipe the response to the client
    const stream = new ReadableStream({
      async start(controller) {
        if (!response.body) {
          controller.close();
          return;
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          controller.enqueue(chunk);
        }
        controller.close();
      },
    });

    return new Response(stream, {
      headers: { "Content-Type": "text/event-stream" },
    });
  } catch (err) {
    console.error(err);
    return json(
      { error: "There was an error processing your request." },
      { status: 500 },
    );
  }
}
