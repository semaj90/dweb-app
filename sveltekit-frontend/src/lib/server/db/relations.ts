import { relations } from "drizzle-orm/relations";
import { cases, citations, users, userSessions, activityLogs, keys } from "./schema";

export const citationsRelations = relations(citations, ({one}) => ({
	case: one(cases, {
		fields: [citations.caseId],
		references: [cases.id]
	}),
}));

export const casesRelations = relations(cases, ({many}) => ({
	citations: many(citations),
}));

export const userSessionsRelations = relations(userSessions, ({one}) => ({
	user: one(users, {
		fields: [userSessions.userId],
		references: [users.id]
	}),
}));

export const usersRelations = relations(users, ({many}) => ({
	userSessions: many(userSessions),
	activityLogs: many(activityLogs),
	keys: many(keys),
}));

export const activityLogsRelations = relations(activityLogs, ({one}) => ({
	user: one(users, {
		fields: [activityLogs.userId],
		references: [users.id]
	}),
}));

export const keysRelations = relations(keys, ({one}) => ({
	user: one(users, {
		fields: [keys.userId],
		references: [users.id]
	}),
}));