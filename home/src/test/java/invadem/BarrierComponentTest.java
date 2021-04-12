package invadem;

import invadem.gameobject.BarrierComponent;
import invadem.gameobject.Projectile;
import org.junit.Test;
import static org.junit.Assert.*;

public class BarrierComponentTest{

    @Test
    public void barrierComponentConstruction() {
        BarrierComponent b = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        assertNotNull(b);
    }
    @Test
    public void barrierComponentGet() {
        BarrierComponent b = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        assertEquals(0, b.getX());
        assertEquals(0, b.getY());
        assertEquals(8,b.getWidth());
        assertEquals(8,b.getHeight());
        assertEquals(3,b.getHealth());
        assertEquals(0,b.getVelocity());
    }

    @Test
    public void testBarrierNotDestroyed() {
        BarrierComponent b = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        assertEquals(true, b.alived());
    }
    @Test
    public void testBarrierDestroyed() {
        BarrierComponent b = new BarrierComponent(null,null,null,null,0,0,8,8,0,0);
        assertEquals(false, b.alived());
    }
    @Test
    public void testBarrierSetHealth() {
        BarrierComponent b = new BarrierComponent(null,null,null,null,0,0,8,8,0,0);
        b.setHealth(3);
        assertEquals(3, b.getHealth());
    }
    @Test
    public void testBarrierCopy() {
        BarrierComponent b = new BarrierComponent(null,null,null,null,0,0,8,8,0,0);
        assertNotNull(b.copy(0,0));
    }

    @Test
    public void testBarrierAttacked(){
        BarrierComponent b = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        assertEquals(3, b.getHealth());
        Projectile p = new Projectile(null,309,464,1,3,1,1 ,1);
        b.attacked(p);
        assertEquals(2, b.getHealth());
    }

//    @Test
//    public void testBarrierHitPointsMax() {
//        Barrier b = /* Your Constructor Here */
//        b.hit();
//        b.hit();
//        assertEquals(1, b.hitPoints());
//    }


//    @Test
//    public void testBarrierIsDestroyed() {
//        Barrier b = /* Your Constructor Here */
//        b.hit();
//        b.hit();
//        b.hit();
//        assertEquals(false, b.isDestroyed());
//    }

}
