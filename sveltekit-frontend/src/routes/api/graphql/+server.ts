import { mcpContext72GetLibraryDocs } from "$lib/mcp-context72-get-library-docs";
import type { RequestHandler } from './$types';


// import { createYoga as createServer } from "graphql-yoga"; // TODO: Install graphql-yoga dependency
// import { makeExecutableSchema } from "@graphql-tools/schema"; // TODO: Install @graphql-tools/schema dependency
const createServer = null as any;
const makeExecutableSchema = null as any;
import { enhancedSearchWithNeo4j } from "$lib/ai/custom-reranker";

const typeDefs = /* GraphQL */ `
  type Recommendation {
    id: ID!
    content: String!
    score: Float!
    intent: String
    timeOfDay: String
    position: String
  }
  type Query {
    recommendations(
      query: String!
      userContext: JSON
      neo4jContext: JSON
      limit: Int
    ): [Recommendation!]!
  }
  scalar JSON
`;

const resolvers = {
  Query: {
    recommendations: async (
      _: any,
      { query, userContext, neo4jContext, limit = 5 }
    ) => {
      const reranked = await enhancedSearchWithNeo4j(
        query,
        userContext,
        neo4jContext,
        limit * 2
      );
      // const memory = await accessMemoryMCP(query, userContext);
      const docs = await mcpContext72GetLibraryDocs("svelte", "runes");
      return reranked
        .map((result) => {
          let score = result.rerankScore;
          // if (memory.some((m) => m.relatedId === result.id)) score += 1;
          // if (docs && docs.includes(result.intent)) score += 1;
          return {
            id: result.id,
            content: result.content,
            score,
            intent: result.intent,
            timeOfDay: result.timeOfDay,
            position: result.position,
          };
        })
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
    },
  },
};

const schema = makeExecutableSchema({ typeDefs, resolvers });

export default createServer({ schema });
