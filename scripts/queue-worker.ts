// Queue Worker for YoRHa Interface
// Processes messages from RabbitMQ queues

import dotenv from 'dotenv';
import { and, eq, sql } from 'drizzle-orm';
import nodemailer from 'nodemailer';
import { db } from '../src/lib/yorha/db';
import { achievements, missions, units, userAchievements, userActivity, userMissions } from '../src/lib/yorha/db/schema';
import { QueueService } from '../src/lib/yorha/services/queue.service';
import { VectorService } from '../src/lib/yorha/services/vector.service';

dotenv.config();

// Initialize services
const queueService = new QueueService();
const vectorService = new VectorService();

// Email transporter (defensive: only create if creds exist)
const emailTransporter = (process.env.SMTP_USER && process.env.SMTP_PASS)
  ? nodemailer.createTransport({
      host: process.env.SMTP_HOST || 'smtp.gmail.com',
      port: parseInt(process.env.SMTP_PORT || '587'),
      secure: false,
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS
      }
    })
  : null;

// Email processor
async function processEmailQueue() {
  console.log('📧 Starting email queue processor...');

  await queueService.consume('email', async (message) => {
    console.log('Processing email:', message.type);

    try {
      switch (message.type) {
        case 'welcome':
          await sendWelcomeEmail(message);
          break;
        case 'password-reset':
          await sendPasswordResetEmail(message);
          break;
        case 'achievement':
          await sendAchievementEmail(message);
          break;
        case 'mission-complete':
          await sendMissionCompleteEmail(message);
          break;
        default:
          console.warn('Unknown email type:', message.type);
      }
    } catch (error) {
      console.error('Email processing error:', error);
      throw error; // Will cause message to be requeued
    }
  }, { prefetch: 5 });
}

// Achievement processor
async function processAchievementQueue() {
  console.log('🏆 Starting achievement queue processor...');

  await queueService.consume('achievements', async (message) => {
    console.log('Processing achievement check:', message.action);

    try {
      switch (message.action) {
        case 'check':
          await checkAchievements(message.userId, message.type);
          break;
        case 'unlock':
          await unlockAchievement(message.userId, message.achievementId);
          break;
        case 'progress':
          await updateAchievementProgress(message.userId, message.achievementId, message.progress);
          break;
        default:
          console.warn('Unknown achievement action:', message.action);
      }
    } catch (error) {
      console.error('Achievement processing error:', error);
      throw error;
    }
  }, { prefetch: 10 });
}

// Activity processor
async function processActivityQueue() {
  console.log('📊 Starting activity queue processor...');

  await queueService.consume('activity', async (message) => {
    console.log('Processing activity:', message.type);

    try {
      // Store activity embedding for semantic search
      if (message.activityId) {
        await vectorService.storeActivityEmbedding(message.activityId);
      }

      // Update user stats based on activity
      await updateUserStats(message.userId, message.type);

      // Check for level up
      await checkLevelUp(message.userId);

      // Publish analytics event
      await queueService.publishMessage('analytics', {
        event: 'user_activity',
        userId: message.userId,
        type: message.type,
        timestamp: message.timestamp
      });
    } catch (error) {
      console.error('Activity processing error:', error);
      throw error;
    }
  }, { prefetch: 20 });
}

// Mission processor
async function processMissionQueue() {
  console.log('🎯 Starting mission queue processor...');

  await queueService.consume('missions', async (message) => {
    console.log('Processing mission:', message.action);

    try {
      switch (message.action) {
        case 'start':
          await startMission(message.userId, message.missionId);
          break;
        case 'complete':
          await completeMission(message.userId, message.missionId, message.score);
          break;
        case 'abandon':
          await abandonMission(message.userId, message.missionId);
          break;
        case 'progress':
          await updateMissionProgress(message.userId, message.missionId, message.progress);
          break;
        case 'generate':
          await generateMissionEmbedding(message.missionId);
          break;
        default:
          console.warn('Unknown mission action:', message.action);
      }
    } catch (error) {
      console.error('Mission processing error:', error);
      throw error;
    }
  }, { prefetch: 10 });
}

// Vector processing queue
async function processVectorQueue() {
  console.log('🔢 Starting vector processing queue...');

  await queueService.consume('vectorProcessing', async (message) => {
    console.log('Processing vector task:', message.type);

    try {
      switch (message.type) {
        case 'user':
          await vectorService.generateUserEmbedding(message.userId);
          break;
        case 'activity':
          await vectorService.storeActivityEmbedding(message.activityId);
          break;
        case 'mission':
          await generateMissionEmbedding(message.missionId);
          break;
        case 'knowledge':
          await vectorService.storeKnowledgeEmbedding(message.content, message.metadata);
          break;
        case 'batch':
          await vectorService.batchUpdateEmbeddings(message.collection, message.items);
          break;
        default:
          console.warn('Unknown vector task type:', message.type);
      }
    } catch (error) {
      console.error('Vector processing error:', error);
      throw error;
    }
  }, { prefetch: 5 });
}

// Helper functions

async function sendWelcomeEmail(data: any) {
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: 'Courier New', monospace; background: #D4D3A7; color: #454138; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { border-bottom: 2px solid #8B8680; padding-bottom: 20px; margin-bottom: 20px; }
        .title { font-size: 24px; font-weight: bold; }
        .content { line-height: 1.6; }
        .button { display: inline-block; padding: 10px 20px; background: #454138; color: #D4D3A7; text-decoration: none; margin: 20px 0; }
        .footer { border-top: 1px solid #8B8680; padding-top: 20px; margin-top: 20px; font-size: 12px; opacity: 0.7; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <div class="title">YoRHa NETWORK</div>
          <div>UNIT ACTIVATION NOTICE</div>
        </div>
        <div class="content">
          <p>Welcome, ${data.unitName}</p>
          <p>Your unit has been successfully registered in the YoRHa Network.</p>
          <p>Please verify your email address to complete the activation process:</p>
          <a href="${process.env.APP_URL}/verify-email?token=${data.verificationToken}" class="button">
            VERIFY EMAIL
          </a>
          <p>Your mission: Fight for the glory of mankind.</p>
        </div>
        <div class="footer">
          <p>FOR THE GLORY OF MANKIND</p>
          <p>YoRHa Network Interface v13.2.7</p>
        </div>
      </div>
    </body>
    </html>
  `;

  if (!emailTransporter) {
    console.warn('Email transporter not configured; skipping welcome email');
    return;
  }
  await emailTransporter.sendMail({
    from: process.env.SMTP_FROM || 'yorha@network.com',
    to: data.to,
    subject: 'YoRHa Unit Activation',
    html
  });

  console.log(`Welcome email sent to ${data.to}`);
}

async function sendPasswordResetEmail(data: any) {
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: 'Courier New', monospace; background: #D4D3A7; color: #454138; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { border-bottom: 2px solid #8B8680; padding-bottom: 20px; margin-bottom: 20px; }
        .title { font-size: 24px; font-weight: bold; }
        .content { line-height: 1.6; }
        .button { display: inline-block; padding: 10px 20px; background: #454138; color: #D4D3A7; text-decoration: none; margin: 20px 0; }
        .footer { border-top: 1px solid #8B8680; padding-top: 20px; margin-top: 20px; font-size: 12px; opacity: 0.7; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <div class="title">YoRHa NETWORK</div>
          <div>ACCESS CODE RESET REQUEST</div>
        </div>
        <div class="content">
          <p>Unit: ${data.unitName}</p>
          <p>A password reset has been requested for your account.</p>
          <p>Click the link below to reset your access code:</p>
          <a href="${process.env.APP_URL}/reset-password?token=${data.resetToken}" class="button">
            RESET ACCESS CODE
          </a>
          <p>This link will expire in 1 hour.</p>
          <p>If you did not request this reset, please ignore this message.</p>
        </div>
        <div class="footer">
          <p>FOR THE GLORY OF MANKIND</p>
          <p>YoRHa Network Interface v13.2.7</p>
        </div>
      </div>
    </body>
    </html>
  `;

  if (!emailTransporter) {
    console.warn('Email transporter not configured; skipping password reset email');
    return;
  }
  await emailTransporter.sendMail({
    from: process.env.SMTP_FROM || 'yorha@network.com',
    to: data.to,
    subject: 'YoRHa Access Code Reset',
    html
  });

  console.log(`Password reset email sent to ${data.to}`);
}

async function sendAchievementEmail(data: any) {
  // Implementation for achievement unlock emails
  console.log('Sending achievement email:', data);
}

async function sendMissionCompleteEmail(data: any) {
  // Implementation for mission completion emails
  console.log('Sending mission complete email:', data);
}

async function checkAchievements(userId: string, type: string) {
  console.log(`Checking achievements for user ${userId}, type: ${type}`);

  // Get user stats
  const user = await db.query.units.findFirst({
    where: eq(units.id, userId)
  });

  if (!user) return;

  // Check various achievement conditions
  const achievementsToCheck = await db.query.achievements.findMany({
    where: eq(achievements.active, true)
  });

  for (const achievement of achievementsToCheck) {
    // Check if already unlocked
    const existing = await db.query.userAchievements.findFirst({
      where: and(
        eq(userAchievements.userId, userId),
        eq(userAchievements.achievementId, achievement.id)
      )
    });

    if (existing?.unlockedAt) continue;

    // Check achievement conditions based on category
    let shouldUnlock = false;
    let progress = 0;

    switch (achievement.category) {
      case 'missions':
        progress = Math.min(100, (user.missionsCompleted / achievement.requiredValue) * 100);
        shouldUnlock = user.missionsCompleted >= achievement.requiredValue;
        break;
      case 'level':
        progress = Math.min(100, (user.level / achievement.requiredValue) * 100);
        shouldUnlock = user.level >= achievement.requiredValue;
        break;
      case 'combat':
        progress = Math.min(100, (parseFloat(user.combatRating) / achievement.requiredValue) * 100);
        shouldUnlock = parseFloat(user.combatRating) >= achievement.requiredValue;
        break;
      case 'time':
        progress = Math.min(100, (user.hoursActive / achievement.requiredValue) * 100);
        shouldUnlock = user.hoursActive >= achievement.requiredValue;
        break;
    }

    if (shouldUnlock) {
      await unlockAchievement(userId, achievement.id);
    } else if (progress > 0) {
      await updateAchievementProgress(userId, achievement.id, progress);
    }
  }
}

async function unlockAchievement(userId: string, achievementId: string) {
  console.log(`Unlocking achievement ${achievementId} for user ${userId}`);

  // Check if already exists
  const existing = await db.query.userAchievements.findFirst({
    where: and(
      eq(userAchievements.userId, userId),
      eq(userAchievements.achievementId, achievementId)
    )
  });

  if (existing) {
    // Update existing record
    await db.update(userAchievements)
      .set({
        unlockedAt: new Date(),
        progress: 100
      })
      .where(eq(userAchievements.id, existing.id));
  } else {
    // Create new record
    await db.insert(userAchievements).values({
      userId,
      achievementId,
      progress: 100,
      unlockedAt: new Date()
    });
  }

  // Get achievement details
  const achievement = await db.query.achievements.findFirst({
    where: eq(achievements.id, achievementId)
  });

  if (achievement) {
    // Award XP
    await db.update(units)
      .set({
        xp: sql`${units.xp} + ${achievement.xpReward}`,
        achievementsUnlocked: sql`${units.achievementsUnlocked} + 1`
      })
      .where(eq(units.id, userId));

    // Log activity
    await db.insert(userActivity).values({
      userId,
      activityType: 'achievement_unlock',
      description: `Unlocked achievement: ${achievement.name}`,
      metadata: { achievementId, xpReward: achievement.xpReward }
    });

    // Send notification
    await queueService.publishMessage('notifications', {
      userId,
      type: 'achievement_unlock',
      title: 'Achievement Unlocked!',
      message: achievement.name,
      icon: achievement.icon
    });
  }
}

async function updateAchievementProgress(userId: string, achievementId: string, progress: number) {
  const existing = await db.query.userAchievements.findFirst({
    where: and(
      eq(userAchievements.userId, userId),
      eq(userAchievements.achievementId, achievementId)
    )
  });

  if (existing) {
    await db.update(userAchievements)
      .set({ progress })
      .where(eq(userAchievements.id, existing.id));
  } else {
    await db.insert(userAchievements).values({
      userId,
      achievementId,
      progress
    });
  }
}

async function updateUserStats(userId: string, activityType: string) {
  console.log(`Updating stats for user ${userId}, activity: ${activityType}`);

  // Update specific stats based on activity type
  switch (activityType) {
    case 'mission_complete':
      await db.update(units)
        .set({
          missionsCompleted: sql`${units.missionsCompleted} + 1`
        })
        .where(eq(units.id, userId));
      break;
    case 'combat_action':
      // Update combat rating based on performance
      break;
    case 'login':
      // Update hours active
      const lastLogin = await db.query.userActivity.findFirst({
        where: and(
          eq(userActivity.userId, userId),
          eq(userActivity.activityType, 'logout')
        ),
        orderBy: (activity, { desc }) => [desc(activity.createdAt)]
      });

      if (lastLogin) {
        const hours = Math.floor((Date.now() - new Date(lastLogin.createdAt).getTime()) / (1000 * 60 * 60));
        await db.update(units)
          .set({
            hoursActive: sql`${units.hoursActive} + ${hours}`
          })
          .where(eq(units.id, userId));
      }
      break;
  }
}

async function checkLevelUp(userId: string) {
  const user = await db.query.units.findFirst({
    where: eq(units.id, userId)
  });

  if (!user) return;

  const maxXp = (user.level * 1000) + 1000;

  if (user.xp >= maxXp) {
    const newLevel = user.level + 1;
    const remainingXp = user.xp - maxXp;

    // Calculate new rank
    let newRank = user.rank;
    if (newLevel >= 50) newRank = 'SS';
    else if (newLevel >= 40) newRank = 'S';
    else if (newLevel >= 30) newRank = 'A';
    else if (newLevel >= 20) newRank = 'B';
    else if (newLevel >= 10) newRank = 'C';
    else if (newLevel >= 5) newRank = 'D';

    await db.update(units)
      .set({
        level: newLevel,
        xp: remainingXp,
        rank: newRank as any
      })
      .where(eq(units.id, userId));

    // Log activity
    await db.insert(userActivity).values({
      userId,
      activityType: 'level_up',
      description: `Advanced to level ${newLevel}`,
      metadata: { newLevel, newRank }
    });

    // Send notification
    await queueService.publishMessage('notifications', {
      userId,
      type: 'level_up',
      title: 'Level Up!',
      message: `You are now level ${newLevel}`,
      icon: '⬆️'
    });

    // Check for level-based achievements
    await queueService.publishMessage('achievements', {
      userId,
      action: 'check',
      type: 'level_up'
    });
  }
}

async function startMission(userId: string, missionId: string) {
  await db.insert(userMissions).values({
    userId,
    missionId,
    status: 'active',
    progress: {}
  });

  await db.insert(userActivity).values({
    userId,
    activityType: 'mission_start',
    description: 'Started new mission',
    metadata: { missionId }
  });
}

async function completeMission(userId: string, missionId: string, score: number) {
  const userMission = await db.query.userMissions.findFirst({
    where: and(
      eq(userMissions.userId, userId),
      eq(userMissions.missionId, missionId),
      eq(userMissions.status, 'active')
    ),
    with: {
      mission: true
    }
  });

  if (!userMission) return;

  const timeTaken = Math.floor((Date.now() - new Date(userMission.startedAt).getTime()) / 1000);

  await db.update(userMissions)
    .set({
      status: 'completed',
      completedAt: new Date(),
      score,
      timeTaken
    })
    .where(eq(userMissions.id, userMission.id));

  // Award rewards
  if (userMission.mission?.rewards) {
    const rewards = userMission.mission.rewards as any;

    if (rewards.xp) {
      await db.update(units)
        .set({
          xp: sql`${units.xp} + ${rewards.xp}`
        })
        .where(eq(units.id, userId));
    }
  }

  await updateUserStats(userId, 'mission_complete');
  await checkLevelUp(userId);
}

async function abandonMission(userId: string, missionId: string) {
  await db.update(userMissions)
    .set({ status: 'abandoned' })
    .where(and(
      eq(userMissions.userId, userId),
      eq(userMissions.missionId, missionId),
      eq(userMissions.status, 'active')
    ));
}

async function updateMissionProgress(userId: string, missionId: string, progress: any) {
  await db.update(userMissions)
    .set({ progress })
    .where(and(
      eq(userMissions.userId, userId),
      eq(userMissions.missionId, missionId),
      eq(userMissions.status, 'active')
    ));
}

async function generateMissionEmbedding(missionId: string) {
  const mission = await db.query.missions.findFirst({
    where: eq(missions.id, missionId)
  });

  if (mission) {
    await vectorService.storeMissionEmbedding(mission);
  }
}

// Main worker function
async function startWorker() {
  console.log('🚀 Starting YoRHa Queue Worker...');
  console.log('====================================');

  try {
    // Start all queue processors
    await Promise.all([
      processEmailQueue(),
      processAchievementQueue(),
      processActivityQueue(),
      processMissionQueue(),
      processVectorQueue()
    ]);

    console.log('✅ All queue processors started successfully');
    console.log('FOR THE GLORY OF MANKIND');

    // Keep the process alive
    process.on('SIGINT', async () => {
      console.log('\n🛑 Shutting down queue worker...');
      await queueService.close();
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      console.log('\n🛑 Shutting down queue worker...');
      await queueService.close();
      process.exit(0);
    });

  } catch (error) {
    console.error('❌ Failed to start queue worker:', error);
    process.exit(1);
  }
}

// Start the worker
startWorker();