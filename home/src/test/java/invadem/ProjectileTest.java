package invadem;

import invadem.gameobject.Invader;
import invadem.gameobject.Projectile;
import invadem.gameobject.Tank;
import org.junit.Test;
import static org.junit.Assert.*;

public class ProjectileTest {


    @Test
    public void testProjectileConstruction() {
        Projectile p = new Projectile(null,309,464,1,3,1,1 ,1);
        assertNotNull(p);
    }
    @Test
    public void testProjectileGet() {
        Projectile p = new Projectile(null,309,464,1,3,1,1 ,1);
        assertEquals(309, p.getX());
        assertEquals(464, p.getY());
        assertEquals(1,p.getWidth());
        assertEquals(3,p.getHeight());
        assertEquals(1,p.getHealth());
        assertEquals(1,p.getVelocity());
        assertEquals(1,p.getDamage());
    }


    @Test
    public void testProjectileIsFriendly() {
        Projectile p = new Projectile(null,309,464,1,3,1,1 ,1);
        assertTrue(p.isFriendly());
    }
    @Test
    public void testProjectileTick() {
        Projectile p = new Projectile(null,309,464,1,3,1,1 ,1);
        p.tick();
        assertEquals(463,p.getY());
    }

    @Test
    public void testProjectileIsNotFriendly() {
        Projectile p = new Projectile(null,309,464,1,3,1,-1 ,1);
        assertFalse(p.isFriendly());
    }
    @Test
    public void testProjectileGetDamage() {
        Projectile p = new Projectile(null,309,464,1,3,1,-1 ,1);
        assertEquals(1,p.getDamage());
    }

    @Test
    public void testProjectileIntersect() {
        Projectile p = new Projectile(null,309,464,1,3,1,1 ,1);
        Invader inv = new Invader(null,null,null,309,464,16,16,1,1,1);
        Tank tank = new Tank(null, null,309, 464, 22, 16, 3,1);
        assertTrue(p.intersect(inv));
        assertTrue(p.intersect(tank));
    }

}
